"""
Implement instanton_exponent, which evaluates the exponent of the instanton tunnelling frequency:
the action of the folded ring-polymer path measured relative to the two wells over the imaginary
times assigned to them.

A tunnelling matrix element decays exponentially with the action of the tunnelling path. A ring
polymer that spends most of its imaginary time resting in the wells accumulates, besides the
tunnelling contribution, the trivial action of a particle sitting at a well minimum. The exponent
that controls tunnelling is the action in excess of this reference.

Returns
-------
float, the instanton exponent S_inst - tau_low*V_low/2 - tau_high*V_high/2 in units of hbar (hbar = 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def instanton_exponent(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> float:
    '''Exponent S_inst - tau_low V_low / 2 - tau_high V_high / 2 of the tunnelling frequency.

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
    exponent : float
        S_inst - tau_low * V_low / 2 - tau_high * V_high / 2 in units of hbar (hbar = 1).
        S_inst is S_half of half_ring_action at the given beads, V_low and V_high are the
        energies of the lower and the higher minimum of locate_stationary_points, and tau_low
        and tau_high are the imaginary times of the two arcs of select_dividing_surface for
        this k.

    Raises
    ------
    ValueError
        If k is not an integer in 1..N/2 - 1, n_beads is not an even integer >= 4, beta is
        not positive, beads does not hold N/2 + 1 finite values, params is invalid, or the
        surface does not have exactly two minima.
    '''
    return exponent

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_instanton_exponent(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> float:
    action = _oracle_half_ring_action(beads, params, beta, n_beads)[0]
    k = _dividing_bead(k, n_beads)
    tau = _ring_step(beta, n_beads)
    v_low, v_high = _oracle_locate_stationary_points(params)[[3, 5]]
    tau_low = 2 * k * tau
    tau_high = (int(n_beads) - 2 * k) * tau
    return float(action - 0.5 * tau_low * v_low - 0.5 * tau_high * v_high)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: instanton of the asymmetric model at beta = 300, N = 1024 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = _oracle_optimize_instanton(params, 300.0, 1024)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 1024)[0])
""",
            "call": "instanton_exponent(beads.copy(), params.copy(), 300.0, 1024, k)",
            "gold_call": "_oracle_instanton_exponent(beads.copy(), params.copy(), 300.0, 1024, k)",
            "tol": 1e-9,
        },
        # --- Typical: the wide right well is the deeper one, at beta = 200 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -1.5, 1.5])
beads = _oracle_optimize_instanton(params, 200.0, 512)
k = int(_oracle_select_dividing_surface(beads, params, 200.0, 512)[0])
""",
            "call": "instanton_exponent(beads.copy(), params.copy(), 200.0, 512, k)",
            "gold_call": "_oracle_instanton_exponent(beads.copy(), params.copy(), 200.0, 512, k)",
            "tol": 1e-9,
        },
        # --- Edge: mirror-symmetric surface (both references equal) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
beads = _oracle_optimize_instanton(params, 300.0, 512)
k = int(_oracle_select_dividing_surface(beads, params, 300.0, 512)[0])
""",
            "call": "instanton_exponent(beads.copy(), params.copy(), 300.0, 512, k)",
            "gold_call": "_oracle_instanton_exponent(beads.copy(), params.copy(), 300.0, 512, k)",
            "tol": 1e-9,
        },
        # --- Boundary: the anchor model on a coarse ring, pinned one bead from the lower turning point ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.0, -0.46, 2.0])
beads = _oracle_optimize_instanton(params, 300.0, 256)
""",
            "call": "instanton_exponent(beads.copy(), params.copy(), 300.0, 256, 1)",
            "gold_call": "_oracle_instanton_exponent(beads.copy(), params.copy(), 300.0, 256, 1)",
            "tol": 1e-9,
        },
        # --- Edge: a straight trial path between the minima with an off-barrier dividing bead ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = np.linspace(-4.2, 4.4, 33)
""",
            "call": "instanton_exponent(beads.copy(), params.copy(), 150.0, 64, 20)",
            "gold_call": "_oracle_instanton_exponent(beads.copy(), params.copy(), 150.0, 64, 20)",
            "tol": 1e-9,
        },
        # --- Invalid: dividing-surface index 0 (the lower turning point) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
beads = np.linspace(-4.2, 4.4, 33)
def run_model():
    try:
        instanton_exponent(beads.copy(), params.copy(), 150.0, 64, 0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_instanton_exponent(beads.copy(), params.copy(), 150.0, 64, 0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

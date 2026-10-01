"""
Implement tunnelling_frequency, the end-to-end pipeline, which returns the tunnelling frequency
hbar*Omega between the two wells of the lower adiabatic surface of the two-state model from
ring-polymer instanton theory for asymmetric wells, at a given inverse temperature and
ring-polymer size.

hbar*Omega is the off-diagonal element of the two-state Hamiltonian that couples the vibrational
ground states localized in the two wells. Together with the asymmetry of the localized energies it
determines the tunnelling splitting of the lowest pair of levels. Ring-polymer instanton theory
evaluates it from a single optimized path, combining the action of the path, its speed at the
dividing surface and the Gaussian fluctuations about it.

Returns
-------
float, the tunnelling frequency hbar*Omega in reduced units
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tunnelling_frequency(params: "np.ndarray", beta: float, n_beads: int) -> float:
    '''Ring-polymer instanton tunnelling frequency hbar*Omega for asymmetric wells.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    hbar_omega : float
        hbar*Omega = hbar * qdot_sigma / (Phi_pin * sqrt(2*pi*hbar)) * exp(-exponent/hbar)
        with hbar = 1, where the instanton comes from optimize_instanton, qdot_sigma and its
        dividing-surface bead k from select_dividing_surface, Phi_pin from fluctuation_factor
        and the exponent from instanton_exponent, all at this k.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential), n_beads is not an even integer
        >= 4, beta is not positive, the surface does not have exactly two minima, or the
        instanton is not found.
    '''
    return hbar_omega

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tunnelling_frequency(params: "np.ndarray", beta: float, n_beads: int) -> float:
    beads = _oracle_optimize_instanton(params, beta, n_beads)
    surface = _oracle_select_dividing_surface(beads, params, beta, n_beads)
    k, qdot = int(surface[0]), float(surface[6])
    phi = float(_oracle_fluctuation_factor(beads, params, beta, n_beads, k)[3])
    exponent = _oracle_instanton_exponent(beads, params, beta, n_beads, k)
    return float(qdot / (phi * np.sqrt(2.0 * np.pi)) * np.exp(-exponent))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: the asymmetric model at beta = 300 with N = 1024 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 1.5])
""",
            "call": "tunnelling_frequency(params.copy(), 300.0, 1024)",
            "gold_call": "_oracle_tunnelling_frequency(params.copy(), 300.0, 1024)",
            "tol": 1e-10,
        },
        # --- Typical: the mirror image (x0 < 0) at a higher temperature, beta = 200, N = 512 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, -4.6, -0.19, 1.5])
""",
            "call": "tunnelling_frequency(params.copy(), 200.0, 512)",
            "gold_call": "_oracle_tunnelling_frequency(params.copy(), 200.0, 512)",
            "tol": 1e-10,
        },
        # --- Edge: high, wide barrier with a small tunnelling frequency (about 3e-6) ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 5.7, -0.29, 2.0])
""",
            "call": "tunnelling_frequency(params.copy(), 300.0, 1024)",
            "gold_call": "_oracle_tunnelling_frequency(params.copy(), 300.0, 1024)",
            "tol": 1e-12,
        },
        # --- Typical: the wide right well is the deeper one ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -1.5, 1.5])
""",
            "call": "tunnelling_frequency(params.copy(), 300.0, 1024)",
            "gold_call": "_oracle_tunnelling_frequency(params.copy(), 300.0, 1024)",
            "tol": 1e-9,
        },
        # --- Boundary: mirror-symmetric surface with degenerate wells, N = 512 ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 1.0, 3.0, 0.0, 1.0])
""",
            "call": "tunnelling_frequency(params.copy(), 300.0, 512)",
            "gold_call": "_oracle_tunnelling_frequency(params.copy(), 300.0, 512)",
            "tol": 1e-10,
        },
        # --- Invalid: coupling so strong that the surface has a single well ---
        {
            "setup": """import numpy as np
params = np.array([1.0, 0.3, 4.6, -0.19, 30.0])
def run_model():
    try:
        tunnelling_frequency(params.copy(), 300.0, 1024)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_tunnelling_frequency(params.copy(), 300.0, 1024)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

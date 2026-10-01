"""
Asymmetry, tunnelling frequency and level splitting of the double well. SPECIFICATION: set eps to tau_inst divided by n_beads, obtain the well array, build the starting trajectory, refine it, and take the crossing data from it, all exactly as in the earlier steps. Then evaluate the action of the refined trajectory, the logarithm of the determinant with the crossing modes removed, and the collapsed logarithm for each well using the frequencies in columns two and three of that well's row, and combine them into the projected-flux instanton expression for the tunnelling frequency Omega of the two-well system. Let d be one half of the second well energy minus the first well energy, the energies being column five of the well array. Return a real array of length three holding, in this order, d, hbar times Omega, and the level splitting of the two wells.

Everything built so far now combines into one number. The exponential of the action carries the bulk of the answer and the determinants supply its prefactor, so the determinants are kept as logarithms and only exponentiated at the very end; forming any of them directly would overflow at these chain lengths. The observable combines the crossing term with the energy difference between the wells, so a system can be dominated by either one, and only when the two are comparable does the result test both halves of the calculation.

Returns
-------
numpy.ndarray of shape (3,) and real dtype.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tunnelling_splitting(a: float, c: float, wy: float, hbar: float, n_beads: int, tau_inst: float) -> "np.ndarray":
    """Asymmetry, tunnelling frequency and level splitting of the double well. SPECIFICATION: set eps to tau_inst divided by n_beads, obtain the well array, build the starting trajectory, refine it, and take the crossing data from it, all exactly as in the earlier steps. Then evaluate the action of the refined trajectory, the logarithm of the determinant with the crossing modes removed, and the collapsed logarithm for each well using the frequencies in columns two and three of that well's row, and combine them into the projected-flux instanton expression for the tunnelling frequency Omega of the two-well system. Let d be one half of the second well energy minus the first well energy, the energies being column five of the well array. Return a real array of length three holding, in this order, d, hbar times Omega, and the level splitting of the two wells.

    Returns
    -------
    numpy.ndarray of shape (3,) and real dtype.

    Raises
    ------
    ValueError: if n_beads is less than two, if tau_inst is not positive and finite, or if any value it forwards to an earlier step is invalid.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_tunnelling_splitting(a: float, c: float, wy: float, hbar: float, n_beads: int, tau_inst: float) -> "np.ndarray":
    mass = 1.0
    n_beads = int(n_beads)
    if n_beads < 2:
        raise ValueError("n_beads must be at least 2")
    if not np.isfinite(tau_inst) or tau_inst <= 0.0:
        raise ValueError("tau_inst must be positive and finite")
    eps = tau_inst / n_beads
    well = _oracle_well_data(a, c, wy, hbar)
    X0 = _oracle_initial_path(a, c, wy, well, n_beads, tau_inst)
    X = _oracle_optimise_instanton(X0, eps, a, c, wy)
    # keep whichever of the two trajectories is closer to stationary; the refined one
    # always is, but a refinement that failed to move must not be used silently
    if np.linalg.norm(_oracle_path_gradient(X, eps, a, c, wy)) > \
       np.linalg.norm(_oracle_path_gradient(X0, eps, a, c, wy)):
        X = X0
    sd = _oracle_surface_data(X, eps, a, c, wy)
    N_l, N_r, qdot = int(sd[1]), int(sd[2]), sd[3]
    tau_l, tau_r = N_l * eps, N_r * eps
    S_inst = _oracle_path_action(X, eps, a, c, wy)
    ld_pin = _oracle_pinned_log_determinant(X, eps, a, c, wy)
    ld_l = _oracle_collapsed_log_determinant(N_l, eps, well[0, 2:4])
    ld_r = _oracle_collapsed_log_determinant(N_r, eps, well[1, 2:4])
    log_phi = 0.5 * np.log(eps / mass) + 0.25 * (ld_pin - ld_l - ld_r)
    # the two well depths, read straight from the potential at the located minima
    depths = _oracle_potential_and_derivatives(well[:, 0:2], a, c, wy)[:, 0]
    expo = -(S_inst - 0.5 * tau_l * depths[0] - 0.5 * tau_r * depths[1]) / hbar
    log_om = -log_phi + np.log(qdot) - 0.5 * np.log(2.0 * np.pi * hbar) + expo
    om = float(np.exp(log_om))
    d = 0.5 * (well[1, 5] - well[0, 5])
    return np.array([d, hbar * om, 2.0 * np.sqrt(d * d + (hbar * om) ** 2)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np',
         "call": 'tunnelling_splitting(5e-10, 1.6, 0.55, 0.04, 1024, 120.0) * 1e9',
         "gold_call": '_oracle_tunnelling_splitting(5e-10, 1.6, 0.55, 0.04, 1024, 120.0) * 1e9'},   # normal: the task's own configuration, in units of 1e-9
        {"setup": 'import numpy as np',
         "call": 'tunnelling_splitting(0.01, 1.0, 0.8, 0.2, 64, 120.0)',
         "gold_call": '_oracle_tunnelling_splitting(0.01, 1.0, 0.8, 0.2, 64, 120.0)'},   # edge: both terms comparable, coarse grid
        {"setup": 'import numpy as np',
         "call": 'tunnelling_splitting(0.01, 1.6, 0.55, 0.25, 64, 120.0)',
         "gold_call": '_oracle_tunnelling_splitting(0.01, 1.6, 0.55, 0.25, 64, 120.0)'},   # edge: crossing term dominant
        {"setup": 'import numpy as np',
         "call": 'tunnelling_splitting(0.02, 1.2, 0.9, 0.2, 64, 120.0)',
         "gold_call": '_oracle_tunnelling_splitting(0.02, 1.2, 0.9, 0.2, 64, 120.0)'},   # edge: asymmetry term dominant
        {"setup": 'import numpy as np',
         "call": 'tunnelling_splitting(0.0, 1.0, 0.8, 0.25, 40, 120.0)',
         "gold_call": '_oracle_tunnelling_splitting(0.0, 1.0, 0.8, 0.25, 40, 120.0)'},   # boundary: symmetric limit, zero asymmetry
        {"setup": 'import numpy as np\ndef _c():\n    try:\n        tunnelling_splitting(5e-10, 1.6, 0.55, 0.04, 1, 120.0)\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_tunnelling_splitting(5e-10, 1.6, 0.55, 0.04, 1, 120.0)\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: fewer than two intervals
    ]

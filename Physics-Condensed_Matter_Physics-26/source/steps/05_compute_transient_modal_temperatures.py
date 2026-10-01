"""
Assemble the transient modal temperature amplitude for every combination of a depth (odd-parity) mode and a lateral (even-parity) mode, at a given elapsed time.

Only mode pairs consisting of one odd-parity (depth) mode and one even-parity (lateral) mode can contribute to the thermal moment computed in the next step, so the odd-parity rows and the even-parity rows of the input eigenbasis (and their matching projection values) must first be separated out; no other combination of parities is physically meaningful here. The dimensionless temperature rise theta(X1, X2, X3, t) in the locally heated (inner) region obeys the heat equation d(theta)/dt = d^2(theta)/dX1^2 + d^2(theta)/dX2^2 + d^2(theta)/dX3^2 + S, with lengths scaled by the thickness h and time by the transverse diffusion time. X2 is the depth coordinate on [-1/2, 1/2] (illuminated face at X2 = -1/2) and X3 the lateral coordinate on [-1/2, 1/2]; on all four faces of the cross-section theta obeys the same Newton-cooling condition as in Step 2. X1 is the stretched coordinate along the beam's axis, centred on the laser spot and unbounded in both directions, and theta vanishes far from the spot. The source S(X1, X2, X3) is exactly the product of three profiles, with no further factor in these scalings: the depth-wise dimensionless intensity profile of Step 3 in X2, the lateral Gaussian profile of Step 3 in X3, and the same Gaussian profile in X1, since the laser spot is circular. The laser is switched on at t = 0 with theta = 0 everywhere, and S is held fixed afterwards.

For an odd-parity mode psi_odd (used in X2) and an even-parity mode psi_even (used in X3), both the unit-norm eigenfunctions of Step 2, the amplitude Theta(t) to return is the coefficient of psi_odd(X2) * psi_even(X3) in the expansion of the axially integrated temperature rise, the integral of theta over X1 from minus to plus infinity. The equation each amplitude obeys, including its decay rate and its source strength in terms of the Step 3 projections and w, must be derived from the heat equation above, and then solved and evaluated at the requested elapsed time t for every odd/even mode-pair combination; neither the equation nor its solution is given here.

Returns
-------
np.ndarray of shape (n_odd, n_even), float: the transient modal temperature amplitude Theta(t) for every combination of an odd-parity mode (rows, in the order they appear among the odd-parity rows of modes) and an even-parity mode (columns, in the order they appear among the even-parity rows of modes), at the given elapsed time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_transient_modal_temperatures(modes: np.ndarray, proj: np.ndarray, w: float, t: float) -> np.ndarray:
    """Transient modal temperature amplitudes Theta(t) for every odd/even mode pair.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis, with the two parity counts not necessarily equal:
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only), with at least one row of
        each parity present.
    proj : np.ndarray
        Shape (N,), as returned by project_heat_source_onto_modes,
        row-aligned with modes, finite.
    w : float
        Dimensionless laser radius, a finite number > 0.
    t : float
        Dimensionless elapsed time since the laser was switched on, a
        finite number >= 0.

    Returns
    -------
    theta : np.ndarray
        Array of shape (n_odd, n_even): the transient modal temperature
        amplitude for each (odd, even) mode-pair combination, with rows
        ordered as the odd-parity rows appear in modes and columns ordered
        as the even-parity rows appear in modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least one row of each parity; if proj does not have one entry per
        row of modes, or contains a non-finite value; if w is not a finite
        number > 0; or if t is not a finite number >= 0.
    """
    return theta  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_transient_modal_temperatures(modes: np.ndarray, proj: np.ndarray,
                                                  w: float, t: float) -> np.ndarray:
    import numpy as np

    modes = np.asarray(modes, dtype=float)
    proj = np.asarray(proj, dtype=float).ravel()

    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape (N, 3)")
    if proj.size != modes.shape[0]:
        raise ValueError("proj must have exactly one entry per row of modes")
    if not np.all(np.isfinite(proj)):
        raise ValueError("proj must contain only finite values")

    parity = modes[:, 2]
    if not np.all(np.isin(parity, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    is_odd = parity == 1.0
    is_even = parity == 0.0
    if not np.any(is_odd) or not np.any(is_even):
        raise ValueError("modes must contain at least one row of each parity")

    if not (isinstance(w, (int, float, np.floating)) and np.isfinite(w) and float(w) > 0.0):
        raise ValueError("w must be a finite number > 0")
    if not (isinstance(t, (int, float, np.floating)) and np.isfinite(t) and float(t) >= 0.0):
        raise ValueError("t must be a finite number >= 0")

    w = float(w)
    t = float(t)

    nu_odd = modes[is_odd, 0]
    a_odd = proj[is_odd]
    nu_even = modes[is_even, 0]
    b_even = proj[is_even]

    ibar = np.outer(a_odd, b_even) * w * np.sqrt(np.pi)
    kappa2 = np.add.outer(nu_odd ** 2, nu_even ** 2)
    theta = (ibar / kappa2) * (1.0 - np.exp(-kappa2 * t))

    return theta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark eigenbasis, projections, and elapsed time ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
proj = np.array([0.97932591, -0.11996899, -0.02654014, 0.01762376])
w, t = 2.0, 0.066
""",
            "call": "compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t)",
            "gold_call": "_oracle_compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t)",
        },
        # --- Valid: a later elapsed time and a different mode set ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.3, 1.10, 0.0],
    [2.8, 1.30, 1.0],
    [6.6, 1.38, 0.0],
    [8.1, 1.39, 1.0],
])
proj = np.array([0.85, -0.09, -0.01, 0.006])
w, t = 0.5, 1.2
""",
            "call": "compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t)",
            "gold_call": "_oracle_compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t)",
        },
        # --- Boundary: the minimum 2-row eigenbasis (one odd, one even) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
proj = np.array([0.9, -0.1])
w, t = 1.0, 0.5
""",
            "call": "digest(compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t))",
            "gold_call": "digest(_oracle_compute_transient_modal_temperatures(modes.copy(), proj.copy(), w, t))",
        },
        # --- Consistency: t = 0 must give exactly zero for every mode pair (ambient initial
        # condition) ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
proj = np.array([0.97932591, -0.11996899, -0.02654014, 0.01762376])
def check(fn):
    theta = np.asarray(fn(modes.copy(), proj.copy(), 2.0, 0.0), dtype=float)
    return int(theta.shape == (2, 2) and np.max(np.abs(theta)) < 1e-12)
""",
            "call": "check(compute_transient_modal_temperatures)",
            "gold_call": "check(_oracle_compute_transient_modal_temperatures)",
        },
        # --- Consistency: at large elapsed time, the transient value must approach the
        # long-time (steady-state) limit Ibar/kappa^2 ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [2.8, 1.30, 1.0],
    [6.5, 1.39, 0.0],
])
proj = np.array([0.9, -0.1, -0.02])
w = 2.0
def check(fn):
    theta_late = np.asarray(fn(modes.copy(), proj.copy(), w, 50.0), dtype=float)
    is_odd = modes[:, 2] == 1.0
    nu_odd, a_odd = modes[is_odd, 0], proj[is_odd]
    nu_even, b_even = modes[~is_odd, 0], proj[~is_odd]
    ibar = np.outer(a_odd, b_even) * w * np.sqrt(np.pi)
    kappa2 = np.add.outer(nu_odd**2, nu_even**2)
    steady = ibar / kappa2
    return int(np.max(np.abs(theta_late - steady)) < 1e-6)
""",
            "call": "check(compute_transient_modal_temperatures)",
            "gold_call": "check(_oracle_compute_transient_modal_temperatures)",
        },
        # --- Invalid: modes contains only one parity (no valid mode pairs exist) ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [6.5, 1.39, 0.0],
])
proj = np.array([0.9, -0.02])
def run_model():
    try:
        compute_transient_modal_temperatures(modes.copy(), proj.copy(), 2.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_transient_modal_temperatures(modes.copy(), proj.copy(), 2.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative elapsed time ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
proj = np.array([0.9, -0.1])
def run_model():
    try:
        compute_transient_modal_temperatures(modes.copy(), proj.copy(), 2.0, -0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_transient_modal_temperatures(modes.copy(), proj.copy(), 2.0, -0.3)
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

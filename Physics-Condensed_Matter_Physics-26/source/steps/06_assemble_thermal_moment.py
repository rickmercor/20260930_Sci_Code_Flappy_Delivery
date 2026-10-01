"""
Assemble the total transient thermal moment M2(t), the first moment of the dimensionless inner temperature field about the beam's mid-plane, from the per-mode moment-arm weights and the transient modal temperature amplitudes.

The moment-arm weights are given row-aligned with the full eigenbasis (one value per row, of either parity), while the transient modal temperatures are given as a two-dimensional array already organized by odd-parity mode (rows) against even-parity mode (columns), in the order those parities appear among the eigenbasis rows. To combine them, the odd-parity weights and the even-parity weights must first be extracted from the full weight array in that same per-parity order, so that the weight vectors line up with the rows and columns of the temperature array respectively.

The quantity to return is the thermal moment M2(t) = Integral X2 * theta dX2 dX3 dX1 of the dimensionless temperature rise theta over the cross-section and over the whole locally heated (inner) region, in the scalings used throughout this pipeline: cross-sectional coordinates and the stretched axial coordinate of the heated region scaled by the thickness h, temperature scaled by the characteristic temperature rise, and X2 the depth coordinate measured from the mid-plane, increasing away from the illuminated face. It is returned exactly in this normalization, with no additional rescaling; the conversion from this moment to a physical slope and angle belongs to the next step.

Returns
-------
float, the total transient thermal moment M2(t), dimensionless, with no additional rescaling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_thermal_moment(modes: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> float:
    """Assemble the total transient thermal moment M2(t) (first moment of the inner temperature field).

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis, with
        at least one row of each parity (the two parity counts need not be
        equal): column 2 is the parity tag (0.0 or 1.0 only), used here only
        to identify which entries of weights belong to which parity.
    weights : np.ndarray
        Shape (N,), as returned by compute_cross_sectional_moment_weights,
        row-aligned with modes, finite.
    theta : np.ndarray
        Shape (n_odd, n_even), as returned by
        compute_transient_modal_temperatures, with rows ordered as the
        odd-parity rows appear in modes and columns ordered as the
        even-parity rows appear in modes, finite.

    Returns
    -------
    m : float
        The total transient thermal moment M2(t) = Integral X2 * theta
        dX2 dX3 dX1 over the cross-section and the heated region, in the
        pipeline's dimensionless scalings, with no additional rescaling.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least one row of each parity; if weights does not have exactly
        one entry per row of modes; if theta is not two-dimensional with
        shape (n_odd, n_even) matching the number of odd-parity and
        even-parity rows in modes; or if any input contains a non-finite
        value.
    """
    return m  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_thermal_moment(modes: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> float:
    import numpy as np

    modes = np.asarray(modes, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape (N, 3)")
    parity = modes[:, 2]
    if not np.all(np.isin(parity, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    is_odd = parity == 1.0
    is_even = parity == 0.0
    if not np.any(is_odd) or not np.any(is_even):
        raise ValueError("modes must contain at least one row of each parity")

    if weights.size != modes.shape[0]:
        raise ValueError("weights must have exactly one entry per row of modes")
    if theta.ndim != 2 or theta.shape != (int(np.sum(is_odd)), int(np.sum(is_even))):
        raise ValueError("theta must have shape (n_odd, n_even) matching modes")
    for name, arr in (("modes", modes), ("weights", weights), ("theta", theta)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")

    x_arm = weights[is_odd]
    plain_avg = weights[is_even]

    # M2(t) = sum over odd/even pairs of Theta_ij * (odd mode's depth moment arm) * (even mode's lateral average).
    m = float(np.sum(theta * np.outer(x_arm, plain_avg)))
    return m

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark modes, weights and transient temperatures ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
modes = np.array([
    [1.10538433, 1.05162999, 0.0],
    [3.52384339, 1.34482657, 1.0],
    [6.49245235, 1.39211673, 0.0],
    [9.56707173, 1.40384712, 1.0],
])
weights = np.array([0.99890181, 0.28515533, -0.04478937, -0.04102900])
theta = np.array([[-1.31687e-02, 5.34700e-04], [1.85870e-04, -7.19670e-06]])
""",
            "call": "digest(assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
            "gold_call": "digest(_oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
        },
        # --- Valid: a smaller 2x2 system in a different mode order ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
modes = np.array([
    [2.8, 1.30, 1.0],
    [1.1, 1.10, 0.0],
    [6.5, 1.38, 0.0],
    [8.1, 1.39, 1.0],
])
weights = np.array([0.5, 0.9, 0.2, -0.1])
theta = np.array([[0.01, -0.002], [0.0005, -0.0001]])
""",
            "call": "digest(assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
            "gold_call": "digest(_oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
        },
        # --- Boundary: a single mode pair (1x1) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
weights = np.array([0.8, 0.4])
theta = np.array([[0.02]])
""",
            "call": "digest(assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
            "gold_call": "digest(_oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
        },
        # --- Boundary: unequal parity counts (3 odd rows, 1 even row) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
    [9.0, 1.40, 1.0],
    [15.0, 1.41, 1.0],
])
weights = np.array([0.9, 0.4, 0.1, -0.05])
theta = np.array([[0.02], [0.01], [-0.005]])
""",
            "call": "digest(assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
            "gold_call": "digest(_oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy()))",
        },
        # --- Consistency: an all-zero temperature array gives an exactly zero moment ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
    [6.5, 1.39, 0.0],
])
weights = np.array([0.9, 0.4, 0.2])
theta = np.zeros((1, 2))
def check(fn):
    return int(float(fn(modes.copy(), weights.copy(), theta.copy())) == 0.0)
""",
            "call": "check(assemble_thermal_moment)",
            "gold_call": "check(_oracle_assemble_thermal_moment)",
        },
        # --- Invalid: theta shape does not match the number of odd/even rows in modes ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
    [6.5, 1.39, 0.0],
])
weights = np.array([0.9, 0.4, 0.2])
theta = np.zeros((2, 2))
def run_model():
    try:
        assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite entry in weights ---
        {
            "setup": """import numpy as np
modes = np.array([
    [1.1, 1.05, 0.0],
    [3.6, 1.35, 1.0],
])
weights = np.array([np.nan, 0.4])
theta = np.array([[0.02]])
def run_model():
    try:
        assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_thermal_moment(modes.copy(), weights.copy(), theta.copy())
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

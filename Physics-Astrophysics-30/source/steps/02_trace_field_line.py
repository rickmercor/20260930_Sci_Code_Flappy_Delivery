"""
Implement trace_field_line, which follows a field line of the region field

from a seed point for a fixed arclength.

The line obeys dx/dl = bhat(x) with bhat the unit vector of the region field

from step 01. Integration uses classical fixed-step fourth-order Runge-Kutta

with n_sub = 8 substeps between consecutive stations, so the result is fully

deterministic. Stations are equally spaced in arclength from 0 to ell_max.



Inputs

------

region_params: dict, as in step 01

seed: (3,) starting position in R_sun

ell_max: float > 0, arclength to follow, R_sun

n_points: int >= 2, number of stations including both ends



Returns

-------

positions: (n_points, 3) float, station positions in R_sun

Returns
-------
positions : np.ndarray     (n_points, 3) station positions in R_sun; positions[0] equals seed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def trace_field_line(region_params: dict, seed: np.ndarray, ell_max: float,
                     n_points: int) -> np.ndarray:
    '''Trace one field line with fixed-step RK4 (8 substeps per interval).

    Parameters
    ----------
    region_params : dict
        Field parameters accepted by evaluate_field (step 01).
    seed : np.ndarray
        (3,) starting position in R_sun.
    ell_max : float
        Positive finite arclength to follow, in R_sun.
    n_points : int
        Number of equally spaced stations, >= 2, including both endpoints.

    Returns
    -------
    positions : np.ndarray
        (n_points, 3) station positions in R_sun; positions[0] equals seed.

    Raises
    ------
    ValueError
        If seed does not have shape (3,), if ell_max is not a positive finite
        number, or if n_points is less than 2. A region_params dict rejected by
        step 01 propagates that ValueError.
    '''
    return positions  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_trace_field_line(region_params: dict, seed: np.ndarray, ell_max: float,
                           n_points: int) -> np.ndarray:
    s = np.asarray(seed, dtype=float)
    if s.shape != (3,):
        raise ValueError("seed must have shape (3,)")
    if not np.isfinite(float(ell_max)) or float(ell_max) <= 0.0:
        raise ValueError("ell_max must be a positive finite number")
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be >= 2")

    n_sub = 8

    def bhat(x):
        B = _oracle_evaluate_field(x[None, :], region_params)[0]
        return B / np.linalg.norm(B)

    dl = float(ell_max) / (n - 1) / n_sub
    out = np.empty((n, 3))
    out[0] = s
    x = s.copy()
    for i in range(1, n):
        for _ in range(n_sub):
            k1 = bhat(x)
            k2 = bhat(x + 0.5 * dl * k1)
            k3 = bhat(x + 0.5 * dl * k2)
            k4 = bhat(x + dl * k3)
            x = x + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i] = x
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: seed on the region axis (the axis is exactly a field line) ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.055, "up": 7.0, "wp": 1.6, "q0": 0.035, "uq": 7.5, "wq": 1.8}
seed = np.array([0.0, 0.0, 1.0])
""",
            "call": "trace_field_line(rp, seed, 8.0, 101)",
            "gold_call": "_oracle_trace_field_line(rp, seed, 8.0, 101)",
        },
        # --- Normal: off-axis seed, the line bends away from the axis ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.12, "up": 3.0, "wp": 1.0, "q0": 0.07, "uq": 3.5, "wq": 1.0}
seed = np.array([0.05, -0.03, 1.0])
""",
            "call": "trace_field_line(rp, seed, 5.0, 81)",
            "gold_call": "_oracle_trace_field_line(rp, seed, 5.0, 81)",
        },
        # --- Boundary: minimum station count n_points = 2 ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
seed = np.array([0.0, 0.0, 1.0])
""",
            "call": "trace_field_line(rp, seed, 0.5, 2)",
            "gold_call": "_oracle_trace_field_line(rp, seed, 0.5, 2)",
        },
        # --- Invalid: non-positive ell_max ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
seed = np.array([0.0, 0.0, 1.0])
def run_model():
    try:
        trace_field_line(rp, seed, 0.0, 11)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_trace_field_line(rp, seed, 0.0, 11)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: seed of wrong shape ---
        {
            "setup": """import numpy as np
rp = {"B0": 3.0, "hB": 0.8, "aB": 1.9, "z_foot": 1.0,
      "p0": 0.0, "up": 0.0, "wp": 1.0, "q0": 0.0, "uq": 0.0, "wq": 1.0}
seed = np.array([0.0, 0.0])
def run_model():
    try:
        trace_field_line(rp, seed, 1.0, 11)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_trace_field_line(rp, seed, 1.0, 11)
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

"""
Recover the curvature, the two well positions and the dissipation threshold of a bistable rate-independent mark from the turning points of one closed hysteresis loop together with the free-energy difference between its two states.

A rate-independent model earns its keep only if its constants can be read back out of an experiment. For a bistable element driven once up and once down, the loop is characterised by a handful of turning points, and each is a different algebraic combination of the constants: the level reached at the ceiling of the drive fixes one combination of curvature and threshold, the level left behind at baseline fixes another, and the accumulated irreversible cost fixes a third because it is the threshold times the total variation, which the geometry of the loop determines. Those three leave a one-parameter family, since nothing in a loop traversed between the same two branches reveals how the two wells are placed relative to one another. The free-energy difference between the two states, measured independently, closes the family and reduces the identification to one scalar equation.

Returns
-------
np.ndarray of shape (4,), float: the four recovered constants in the order k, a, b, rho, that is the curvature k of both wells, the distance a from the barrier to the bottom of the repressed well, the distance b from the barrier to the bottom of the active well and the dissipation threshold rho. Not a scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def identify_material_parameters(q_peak: float, q_residual: float,
                                 dissipation: float, delta: float,
                                 ell_max: float) -> np.ndarray:
    """Recover the landscape and the dissipation threshold from a loop record.

    The record is produced by a mark that starts at the bottom of the
    repressed well, is carried once past the barrier onto the active branch
    while the interaction potential rises to ell_max, and is then unloaded to
    zero potential without returning across the barrier. Over such a loop the
    peak state is b + (ell_max - rho) / k, the residual state is b + rho / k,
    and the accumulated dissipation is rho times the total variation of the
    trajectory. The free energy of the active well relative to the repressed
    one is delta = k * (a ** 2 - b ** 2) / 2, which follows from insisting
    that the free energy be continuous at the barrier.

    Parameters
    ----------
    q_peak : float
        Largest state reached over the loop.
    q_residual : float
        State left at the end of the loop, at zero interaction potential;
        0 < q_residual < q_peak.
    dissipation : float
        Accumulated dissipation over the loop, dissipation > 0.
    delta : float
        Free energy of the active well above the repressed one.
    ell_max : float
        Ceiling of the interaction potential over the loop, ell_max > 0.

    Returns
    -------
    parameters : np.ndarray
        Array of shape (4,) holding, in order, the curvature k, the distance a
        from the barrier to the bottom of the repressed well, the distance b
        from the barrier to the bottom of the active well, and the dissipation
        threshold rho.

    Raises
    ------
    ValueError
        If q_peak, q_residual, dissipation, delta or ell_max is not a finite
        number, if the record does not satisfy 0 < q_residual < q_peak, if
        dissipation or ell_max is not strictly positive, or if the record
        admits no bistable landscape carrying the given offset.
    """
    return parameters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_identify_material_parameters(q_peak: float, q_residual: float,
                                         dissipation: float, delta: float,
                                         ell_max: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_peak", q_peak), ("q_residual", q_residual),
                        ("dissipation", dissipation), ("delta", delta),
                        ("ell_max", ell_max)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    peak, residual = float(q_peak), float(q_residual)
    cost, offset, ceiling = float(dissipation), float(delta), float(ell_max)
    if not peak > residual > 0.0:
        raise ValueError("the record must satisfy 0 < q_residual < q_peak")
    if cost <= 0.0 or ceiling <= 0.0:
        raise ValueError("dissipation and ell_max must be strictly positive")

    span = peak - residual

    # Every constant is a function of the threshold alone: the two levels fix
    # the curvature and the far well, the irreversible cost fixes the near
    # well, and the free-energy difference is what is left to satisfy.
    def _unfold(threshold):
        curvature = (ceiling - 2.0 * threshold) / span
        if not curvature > 0.0:
            return None
        near = cost / threshold - 2.0 * peak + residual
        far = residual - threshold / curvature
        return curvature, near, far

    def _residual(threshold):
        unfolded = _unfold(threshold)
        if unfolded is None:
            return -np.inf
        curvature, near, far = unfolded
        return 0.5 * curvature * (near * near - far * far) - offset

    # Only thresholds below all three of these leave a landscape that is a
    # landscape at all: beyond them the curvature, the near well or the far
    # well would have to be non-positive.
    upper = min(0.5 * ceiling,
                cost / (2.0 * peak - residual),
                residual * ceiling / (span + 2.0 * residual)) * (1.0 - 1.0e-12)
    lower = 1.0e-12 * upper

    # A vanishing threshold makes the near well arbitrarily wide, so the
    # residual opens positive; the identification is the first crossing, the
    # later ones belonging to landscapes the admissible interval excludes.
    grid = lower + (upper - lower) * np.linspace(0.0, 1.0, 4097)
    values = np.array([_residual(float(point)) for point in grid], dtype=float)
    bracket = None
    for index in range(grid.size - 1):
        if values[index] > 0.0 >= values[index + 1]:
            bracket = (float(grid[index]), float(grid[index + 1]))
            break
    if bracket is None:
        raise ValueError("the record admits no bistable landscape with this offset")

    low, high = bracket
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _residual(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low <= 1.0e-16 * (1.0 + high):
            break

    threshold = 0.5 * (low + high)
    curvature, near, far = _unfold(threshold)
    if not (near > 0.0 and far > 0.0):
        raise ValueError("the recovered landscape is not bistable")

    return np.array([curvature, near, far, threshold], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark record, whose landscape is asymmetric ---
        {
            "setup": """import numpy as np
def digest(block):
    flat = np.ravel(np.asarray(block, dtype=float))
    return float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
record = (0.448487, 0.165654, 0.0835480, 0.00603213, 0.49663102650045726)
""",
            "call": "digest(identify_material_parameters(*record))",
            "gold_call": "digest(_oracle_identify_material_parameters(*record))",
        },
        # --- Valid: a symmetric landscape, recognised by a vanishing offset ---
        {
            "setup": """import numpy as np
def digest(block):
    flat = np.ravel(np.asarray(block, dtype=float))
    return float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
k, a, b, rho, Lm = 1.0, 0.15, 0.15, 0.10, 0.49663102650045726
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = (peak, res, cost, 0.5 * k * (a * a - b * b), Lm)
""",
            "call": "digest(identify_material_parameters(*record))",
            "gold_call": "digest(_oracle_identify_material_parameters(*record))",
        },
        # --- Valid: a stiffer landscape under a lower ceiling ---
        {
            "setup": """import numpy as np
def digest(block):
    flat = np.ravel(np.asarray(block, dtype=float))
    return float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
k, a, b, rho, Lm = 2.4, 0.22, 0.09, 0.055, 0.40
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = (peak, res, cost, 0.5 * k * (a * a - b * b), Lm)
""",
            "call": "digest(identify_material_parameters(*record))",
            "gold_call": "digest(_oracle_identify_material_parameters(*record))",
        },
        # --- Valid: an inverted asymmetry, the active well the deeper one ---
        {
            "setup": """import numpy as np
def digest(block):
    flat = np.ravel(np.asarray(block, dtype=float))
    return float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
k, a, b, rho, Lm = 1.15, 0.11, 0.19, 0.082, 0.49663102650045726
peak = b + (Lm - rho) / k
res = b + rho / k
cost = rho * (a + 2.0 * peak - res)
record = (peak, res, cost, 0.5 * k * (a * a - b * b), Lm)
""",
            "call": "digest(identify_material_parameters(*record))",
            "gold_call": "digest(_oracle_identify_material_parameters(*record))",
        },
        # --- Edge: the record quoted to only four significant figures ---
        {
            "setup": """import numpy as np
def digest(block):
    flat = np.ravel(np.asarray(block, dtype=float))
    return float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
record = (0.4485, 0.1657, 0.08355, 0.006032, 0.49663102650045726)
""",
            "call": "digest(identify_material_parameters(*record))",
            "gold_call": "digest(_oracle_identify_material_parameters(*record))",
        },
        # --- Invalid: a residual level above the peak level ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        identify_material_parameters(0.16, 0.44, 0.08, 0.006, 0.4966)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_identify_material_parameters(0.16, 0.44, 0.08, 0.006, 0.4966)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a loop that dissipates nothing ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        identify_material_parameters(0.448487, 0.165654, 0.0, 0.006032, 0.4966)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_identify_material_parameters(0.448487, 0.165654, 0.0, 0.006032, 0.4966)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an offset so deeply negative that no landscape
        # consistent with the loop can carry it ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        identify_material_parameters(0.448487, 0.165654, 0.0835480, -5.0, 0.49663102650045726)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_identify_material_parameters(0.448487, 0.165654, 0.0835480, -5.0, 0.49663102650045726)
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

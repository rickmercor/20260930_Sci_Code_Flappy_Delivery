"""
Build the propagator of the semi-discrete thermal system over one interval by the precise integration algorithm, with the increment held apart from the identity.

After the far field has been condensed onto the interface and the capacity operator inverted against the conductance operator, the temperature field obeys a linear first-order system whose homogeneous solution over an interval is the exponential of the state matrix times that interval. Evaluating that exponential by a truncated series applied directly to the full interval is hopeless for a diffusion operator, whose eigenvalues span several orders of magnitude, because the series must resolve the fastest mode while the interval is chosen for the slowest. The precise integration algorithm removes the difficulty by subdividing the interval into a power-of-two number of parts, so that the exponential of one part is accurately represented by a very short Taylor series, and then recovering the exponential of the whole interval by squaring that result once per level of subdivision.




The delicacy of the algorithm is entirely in how the intermediate result is stored. After subdivision the exponential of one part differs from the identity by a matrix whose entries are smaller than the identity by many orders of magnitude, and forming the sum explicitly discards most of the significant digits of the small part in the rounding, after which squaring propagates the damaged quantity rather than the intended one. The remedy is to keep only the increment, the exponential minus the identity, and to square in terms of it: the square of the identity plus an increment is the identity plus twice the increment plus the increment squared, so the doubling recursion never adds a tiny matrix to a unit one. The identity is restored only at the very end, when the accumulated increment has grown to a magnitude comparable with it. Getting the number of doublings right matters as much as the recursion itself, since one doubling too few returns the propagator of half the interval, a plausible-looking matrix that silently halves the whole time march.

The result is a propagator that is exact to working precision, not merely accurate to some order in the interval, and this holds however stiff the system is; it is unconditionally stable and introduces no algorithmic damping. That should not be mistaken for exactness of the whole time march, however. The propagator is only the homogeneous part of the solution, and any load acting during the interval enters through a convolution against the same propagator that has to be evaluated by quadrature; the accuracy of that quadrature, not of the propagator, is what limits the step size that can be taken.

Returns
-------
np.ndarray of shape (n, n), float: the propagator of the semi-discrete thermal system over the requested interval.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_precise_propagator(state_matrix: np.ndarray, interval: float,
                               n_levels: int) -> np.ndarray:
    """Build the propagator of a linear first-order system over one interval.

    Parameters
    ----------
    state_matrix : np.ndarray
        Square array of shape (n, n) holding the state matrix of the
        semi-discrete system, in s^-1.
    interval : float
        Length of the interval to propagate over, in seconds
        (interval >= 0).
    n_levels : int
        Number of subdivision levels, so the interval is split into
        2 ** n_levels parts (n_levels >= 0).

    Returns
    -------
    propagator : np.ndarray
        Array of shape (n, n) holding the propagator of the system over the
        given interval.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return propagator  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_compute_precise_propagator(state_matrix: np.ndarray, interval: float,
                                       n_levels: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_state_matrix(state_matrix):
        """Raise ValueError unless the argument is a finite square array."""
        array = np.asarray(state_matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("state_matrix must be a square two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError("state_matrix must have at least one row")
        if not np.all(np.isfinite(array)):
            raise ValueError("state_matrix must contain only finite entries")
        return array

    array = _validate_state_matrix(state_matrix)
    if not (isinstance(interval, (int, float)) and np.isfinite(interval)
            and float(interval) >= 0.0):
        raise ValueError("interval must be a finite number >= 0")
    if not (isinstance(n_levels, (int, np.integer)) and not isinstance(n_levels, bool)
            and int(n_levels) >= 0):
        raise ValueError("n_levels must be an integer >= 0")

    interval = float(interval)
    n_levels = int(n_levels)

    # Exponential of one subdivided part, as an increment on the identity.
    part = array * (interval / 2.0 ** n_levels)
    square = part @ part
    increment = part + 0.5 * square + (square @ part) / 6.0

    # Doubling recursion carried out on the increment alone, so that a quantity
    # far smaller than unity is never added to the identity until the end.
    for _ in range(n_levels):
        increment = 2.0 * increment + increment @ increment

    return np.eye(array.shape[0]) + increment

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================




def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: stiff diffusion-like state matrix over one benchmark step (normal scenario) ---
        {
            "setup": """import numpy as np
n = 12
diag = np.linspace(1.0e-5, 3.0e-3, n)
off = 0.25 * np.diag(np.linspace(1.0e-5, 5.0e-4, n - 1), 1)
state = -(np.diag(diag) + off + off.T)
interval = 2500.0
n_levels = 20

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_precise_propagator(state, interval, n_levels))",
            "gold_call": "digest(_oracle_compute_precise_propagator(state, interval, n_levels))",
        },
        # --- Valid: non-symmetric state matrix over a Gauss sub-interval ---
        {
            "setup": """import numpy as np
state = np.array([[-2.0e-3, 4.0e-4, 0.0],
                  [1.0e-4, -1.5e-3, 3.0e-4],
                  [0.0, 7.0e-5, -9.0e-4]])
interval = 2500.0 * 0.5 * (1.0 - 0.9324695142031521)
n_levels = 20

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_precise_propagator(state, interval, n_levels))",
            "gold_call": "digest(_oracle_compute_precise_propagator(state, interval, n_levels))",
        },
        # --- Boundary: zero interval, so the propagator is the identity ---
        {
            "setup": """import numpy as np
state = np.array([[-1.0e-3, 2.0e-4], [2.0e-4, -5.0e-4]])
interval = 0.0
n_levels = 20

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_precise_propagator(state, interval, n_levels))",
            "gold_call": "digest(_oracle_compute_precise_propagator(state, interval, n_levels))",
        },
        # --- Edge: interval long enough to damp the fast mode but not the slow one ---
        {
            "setup": """import numpy as np
state = np.array([[-1.0e-2, 1.0e-5], [1.0e-5, -2.0e-5]])
interval = 5.0e4
n_levels = 20

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_precise_propagator(state, interval, n_levels))",
            "gold_call": "digest(_oracle_compute_precise_propagator(state, interval, n_levels))",
        },
        # --- Invalid: non-square state matrix ---
        {
            "setup": """import numpy as np
bad = np.zeros((3, 4))
def run_model():
    try:
        compute_precise_propagator(bad, 2500.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_precise_propagator(bad, 2500.0, 20)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative number of subdivision levels ---
        {
            "setup": """import numpy as np
state = np.array([[-1.0e-3, 0.0], [0.0, -2.0e-3]])
def run_model():
    try:
        compute_precise_propagator(state, 2500.0, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_precise_propagator(state, 2500.0, -1)
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

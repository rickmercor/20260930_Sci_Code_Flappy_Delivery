"""
Determine the coefficients of the three-pole Pade approximant of the kinetic response by forcing it to reproduce prescribed response values at two prescribed phase speeds.

A three-pole Pade approximant of the kinetic response carries three coefficients, one of which is already fixed by matching one order beyond the leading fluid asymptote, leaving two free. Requiring the approximant to pass through the response at two prescribed phase speeds is linear in those two, so the anchoring is one small complex solve rather than an optimisation.

Returns
-------
np.ndarray of shape (3,), complex: the numerator, linear denominator and quadratic denominator coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_matched_pade_coefficients(phase_speeds: np.ndarray,
                                    response_values: np.ndarray) -> np.ndarray:
    """Fix the Pade coefficients from two prescribed response values.

    The approximant is the three-pole rational form in the normalised phase
    speed whose numerator and denominator both equal one at zero phase speed,
    whose denominator has degree three with its cubic coefficient tied to the
    numerator coefficient, and whose quadratic denominator coefficient is held
    at minus two. The two remaining coefficients are fixed by requiring the
    approximant to take the given response value at the corresponding phase
    speed.

    Parameters
    ----------
    phase_speeds : np.ndarray
        Complex array of shape (2,) holding the two distinct normalised phase
        speeds at which the approximant is anchored.
    response_values : np.ndarray
        Complex array of shape (2,) holding the value the approximant must take
        at the corresponding entry of ``phase_speeds``.

    Returns
    -------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding, in order, the numerator
        coefficient, the linear denominator coefficient and the quadratic
        denominator coefficient of the approximant.

    Raises
    ------
    ValueError
        If either input is not a finite array of shape (2,), if the two entries
        of ``phase_speeds`` are equal, or if the two anchoring conditions do not
        determine the coefficients.
    """
    return pade_coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_matched_pade_coefficients(phase_speeds: np.ndarray,
                                            response_values: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    speeds = np.asarray(phase_speeds, dtype=complex)
    values = np.asarray(response_values, dtype=complex)
    for name, array in (("phase_speeds", speeds), ("response_values", values)):
        if array.shape != (2,):
            raise ValueError(f"{name} must be an array of shape (2,)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if speeds[0] == speeds[1]:
        raise ValueError("phase_speeds must hold two distinct phase speeds")

    # The approximant is (1 + a z) / (1 + b z + c z**2 - 2 a z**3) with c = -2.
    # Clearing the denominator against the target value R at the anchor z makes
    # the condition linear in a and b:
    #     a (z + 2 R z**3) - b (R z) = -(1 - R - c R z**2).
    quadratic = -2.0 + 0.0j
    matrix = np.empty((2, 2), dtype=complex)
    matrix[:, 0] = speeds + 2.0 * values * speeds ** 3
    matrix[:, 1] = -values * speeds
    rhs = -(1.0 - values - quadratic * values * speeds ** 2)

    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    scale = np.max(np.abs(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= 1.0e-13 * max(scale ** 2, 1.0):
        raise ValueError("the anchoring conditions do not determine the coefficients")

    numerator, linear = np.linalg.solve(matrix, rhs)
    return np.array([numerator, linear, quadratic], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the conjugate-symmetric kinetic root pair at the benchmark
        #     wave number, anchored on the dispersion relation there (normal) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
root = 2.2716812436 - 0.1168988200j
phase_speeds = np.array([root, -np.conj(root)])
response_values = np.full(2, -0.16 + 0.0j)
""",
            "call": "sig(solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
            "gold_call": "sig(_oracle_solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
        },
        # --- Valid: a weakly damped long-wavelength pair, where the coefficients
        #     grow by three orders of magnitude ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
root = 3.7617527119 - 0.0001948341j
phase_speeds = np.array([root, -np.conj(root)])
response_values = np.full(2, -0.04 + 0.0j)
""",
            "call": "sig(solve_matched_pade_coefficients(phase_speeds, response_values), 1000.0)",
            "gold_call": "sig(_oracle_solve_matched_pade_coefficients(phase_speeds, response_values), 1000.0)",
        },
        # --- Valid: a strongly damped short-wavelength pair ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
root = 1.4466732000 - 0.6019815400j
phase_speeds = np.array([root, -np.conj(root)])
response_values = np.full(2, -1.0 + 0.0j)
""",
            "call": "sig(solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
            "gold_call": "sig(_oracle_solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
        },
        # --- Valid: an asymmetric pair with unequal response values, which breaks
        #     the parity that makes the coefficients purely imaginary ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
phase_speeds = np.array([1.4 - 0.6j, -0.9 - 0.25j])
response_values = np.array([-0.35 + 0.12j, 0.44 - 0.03j])
""",
            "call": "sig(solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
            "gold_call": "sig(_oracle_solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
        },
        # --- Boundary: a vanishing response at one anchor, which strips every
        #     term carrying the response out of that condition ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
phase_speeds = np.array([2.0 - 0.1j, -2.0 - 0.1j])
response_values = np.array([0.0 + 0.0j, -0.25 + 0.0j])
""",
            "call": "sig(solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
            "gold_call": "sig(_oracle_solve_matched_pade_coefficients(phase_speeds, response_values), 1.0)",
        },
        # --- Invalid: two coincident anchors, which cannot fix two coefficients ---
        {
            "setup": """import numpy as np
phase_speeds = np.array([1.5 - 0.2j, 1.5 - 0.2j])
response_values = np.full(2, -0.2 + 0.0j)
def run_model():
    try:
        solve_matched_pade_coefficients(phase_speeds, response_values)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_matched_pade_coefficients(phase_speeds, response_values)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a wrongly shaped anchor list ---
        {
            "setup": """import numpy as np
phase_speeds = np.array([1.5 - 0.2j, -1.5 - 0.2j, 0.5 + 0.0j])
response_values = np.full(3, -0.2 + 0.0j)
def run_model():
    try:
        solve_matched_pade_coefficients(phase_speeds, response_values)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_matched_pade_coefficients(phase_speeds, response_values)
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

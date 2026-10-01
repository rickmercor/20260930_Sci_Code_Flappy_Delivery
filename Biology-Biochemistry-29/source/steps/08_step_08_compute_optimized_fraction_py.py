"""
Compose the earlier steps to calibrate the four kinetic rates, construct the calibrated equilibrium model, maximize absolute logarithmic sensitivity, and return the fraction at the resulting admissible equilibrium.

The final scalar depends on the shared inverse parameters, the physically admissible forward equilibrium, and the global sensitivity optimum. Calibration experiments are not the requested final condition. Reuse the previous public functions rather than reimplementing the entire pipeline; the final oracle composes the corresponding earlier oracle functions.

Returns
-------
A finite Python float representing the fraction at the globally sensitivity-maximizing equilibrium.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_optimized_fraction(
    fixed_rates: np.ndarray,
    observations: np.ndarray,
    bounds: np.ndarray,
    T1: float = 1.35,
    T2_lower: float = 0.2,
    T2_upper: float = 2.0,
) -> float:
    """Return the fraction at the globally sensitivity-maximizing equilibrium.

    Inputs are the fixed-rate vector, four calibration observations,
    parameter bounds, and optional final-condition T1 and T2 interval.
    The defaults are T1=1.35, T2_lower=0.2, T2_upper=2.0.

    Compose the previous public functions to calibrate the four
    unknown kinetic rates, construct the calibrated equilibrium
    model, and maximize absolute logarithmic sensitivity. Return
    the fraction evaluated at the resulting admissible optimum.

    Returns:
        float: The finite optimized steady-state fraction.

    Raises:
        ValueError: If any input is invalid, calibration has no
            unique admissible solution, or the optimization interval
            contains no admissible equilibrium. Propagate the
            relevant ValueError from the preceding steps.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial import Polynomial

def _assemble_rates(base, unknown):
    k = np.asarray(base, dtype=float).copy()
    k[[2, 5, 7, 11]] = unknown
    return k

def _oracle_compute_optimized_fraction(
    fixed_rates, observations, bounds,
    T1=1.35, T2_lower=0.2, T2_upper=2.0
):
    import numpy as np
    from numpy.polynomial import Polynomial

    _context = {
        "np": np,
        "Polynomial": Polynomial,
        "_assemble_rates": _assemble_rates,
        "_oracle_compute_equilibrium_coefficients":
            _oracle_compute_equilibrium_coefficients,
        "_oracle_solve_positive_equilibrium":
            _oracle_solve_positive_equilibrium,
        "_oracle_compute_fraction_sensitivity":
            _oracle_compute_fraction_sensitivity,
        "_oracle_construct_inverse_constraints":
            _oracle_construct_inverse_constraints,
        "_oracle_reconstruct_kinetic_parameters":
            _oracle_reconstruct_kinetic_parameters,
        "_oracle_calibrate_kinetic_parameters":
            _oracle_calibrate_kinetic_parameters,
        "_oracle_optimize_sensitivity":
            _oracle_optimize_sensitivity
    }

    for fn in (
        _oracle_compute_equilibrium_coefficients,
        _oracle_solve_positive_equilibrium,
        _oracle_compute_fraction_sensitivity,
        _oracle_construct_inverse_constraints,
        _oracle_reconstruct_kinetic_parameters,
        _oracle_calibrate_kinetic_parameters,
        _oracle_optimize_sensitivity
    ):
        fn.__globals__.update(_context)

    unknown = _oracle_calibrate_kinetic_parameters(
        fixed_rates, observations, bounds
    )

    k = _assemble_rates(fixed_rates, unknown)
    c = _oracle_compute_equilibrium_coefficients(k)

    return float(
        _oracle_optimize_sensitivity(
            c, T1, T2_lower, T2_upper
        )[1]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np

def capture(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
    except Exception:
        return 2.0

base = np.array([
    1.7, 0.83, 1.29, 0.47, 1.11, 1e13,
    0.62, 0.91, 1.37, 0.58, 1.43, 0.76,
    0.66, 1.21
], dtype=float)

observations = np.array([
    0.96334092858302567,
    0.5405066201174169,
    0.53608216437829215,
    -0.070312225957344945
], dtype=float)

alt_observations = np.array([
    0.83422055242381599,
    0.43018961608261541,
    0.42629891243239881,
    -0.055267295833012202
], dtype=float)

bounds = np.array([
    [0.7,1.8],
    [1.0,5.0],
    [0.4,1.2],
    [0.5,1.6]
], dtype=float)
"""

    return [
        {
            "description": "Complete supplied inverse calibration and optimization.",
            "setup": common,
            "call": "compute_optimized_fraction(base,observations,bounds)",
            "gold_call": "_oracle_compute_optimized_fraction(base,observations,bounds)",
            "tol": 1e-9
        },
        {
            "description": "Independent synthetic calibration and optimization.",
            "setup": common,
            "call": "compute_optimized_fraction(base,alt_observations,bounds)",
            "gold_call": "_oracle_compute_optimized_fraction(base,alt_observations,bounds)",
            "tol": 1e-9
        },
        {
            "description": "Invalid parameter bounds propagate ValueError.",
            "setup": common + """
bad = bounds.copy()
bad[0] = [1.5,0.7]
""",
            "call": "capture(compute_optimized_fraction,base,observations,bad)",
            "gold_call": "capture(_oracle_compute_optimized_fraction,base,observations,bad)",
            "tol": 1e-9
        }
    ]

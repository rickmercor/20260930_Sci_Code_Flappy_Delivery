"""
Compute the symmetry-sector energy-center shift, the minimized shifted spectral norm, the optimized finite-difference time shift, and the two coefficient-dependent constants that balance finite-sampling and truncation perturbations. Evaluate the optimized time shift through logarithms so that a representable result remains finite even when direct powers of the spectral norm would overflow.

The sampling perturbation of the projected Hamiltonian scales inversely with the finite-difference time shift, whereas the degree-$J$ truncation perturbation scales as its `2J`-th power. Their positive convex upper bound has a unique stationary point. Centering a known symmetry-sector spectrum at zero first replaces its restricted spectral norm by half its spectral range, reducing the optimized sampling cost before the time-shift balance is evaluated.

Returns
-------
np.ndarray of shape (5,) containing [energy_shift, shifted_spectral_norm, optimized_delta_t, alpha_nJ, beta_nJ] as native float64 values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Integral, Real

import numpy as np

def compute_msd_preprocessing(
    first_derivative_coefficients: np.ndarray,
    krylov_dimension: int,
    degree: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
) -> np.ndarray:
    """Compute the energy shift and optimized time-shift data.

    Parameters
    ----------
    first_derivative_coefficients : np.ndarray
        Real centered first-derivative weights of length ``2 * degree + 1``.
    krylov_dimension : int
        Positive Krylov dimension ``n``.
    degree : int
        Positive central-difference degree ``J``.
    total_shots : float
        Positive finite total shot count ``M``.
    sector_minimum : float
        Finite lower energy bound for the symmetry sector.
    sector_maximum : float
        Finite upper energy bound, strictly larger than ``sector_minimum``.

    Returns
    -------
    preprocessing : np.ndarray
        Float64 array ``[shift, shifted_norm, delta_t, alpha_nJ, beta_nJ]``.

    Raises
    ------
    ValueError
        If an input has an invalid type, shape, range, or non-finite value, or
        if the optimized result is not representable as finite float64.
    """
    return np.empty(5, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_msd_preprocessing(
    first_derivative_coefficients: np.ndarray,
    krylov_dimension: int,
    degree: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real

    import numpy as np

    if isinstance(krylov_dimension, bool) or not isinstance(krylov_dimension, Integral) or int(krylov_dimension) < 1:
        raise ValueError("krylov_dimension must be an integer >= 1")
    if isinstance(degree, bool) or not isinstance(degree, Integral) or int(degree) < 1:
        raise ValueError("degree must be an integer >= 1")
    if isinstance(total_shots, bool) or not isinstance(total_shots, Real):
        raise ValueError("total_shots must be a positive finite real scalar")
    if isinstance(sector_minimum, bool) or not isinstance(sector_minimum, Real):
        raise ValueError("sector_minimum must be finite")
    if isinstance(sector_maximum, bool) or not isinstance(sector_maximum, Real):
        raise ValueError("sector_maximum must be finite")

    degree = int(degree)
    n = int(krylov_dimension)
    shots = float(total_shots)
    e_min = float(sector_minimum)
    e_max = float(sector_maximum)
    if not math.isfinite(shots) or shots <= 0.0:
        raise ValueError("total_shots must be > 0 and finite")
    if not math.isfinite(e_min) or not math.isfinite(e_max) or not e_min < e_max:
        raise ValueError("sector bounds must be finite and strictly ordered")

    raw = np.asarray(first_derivative_coefficients)
    if raw.ndim != 1 or raw.size != 2 * degree + 1:
        raise ValueError("coefficient length must equal 2 * degree + 1")
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError("first-derivative coefficients must be real")
    try:
        coefficients = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("first-derivative coefficients must be real numeric values") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("first-derivative coefficients must be finite")

    half_min = 0.5 * e_min
    half_max = 0.5 * e_max
    energy_shift = half_min + half_max
    shifted_norm = half_max - half_min
    if not math.isfinite(energy_shift) or not math.isfinite(shifted_norm) or shifted_norm <= 0.0:
        raise ValueError("the centered sector quantities must be finite")

    nodes = np.arange(-degree, degree + 1, dtype=np.float64)
    coefficient_norm = float(np.sum(np.abs(coefficients)))
    alpha_nj = 2.0 * n * math.sqrt(2.0 * math.log(2.0 * n)) * coefficient_norm
    factorial = float(math.factorial(2 * degree + 1))
    beta_nj = n * float(np.sum(np.abs(coefficients * nodes ** (2 * degree + 1)))) / factorial
    if not math.isfinite(alpha_nj) or not math.isfinite(beta_nj) or alpha_nj <= 0.0 or beta_nj <= 0.0:
        raise ValueError("the supplied coefficients do not define positive finite balance constants")

    exponent = 2 * degree + 1
    log_delta = (
        math.log(alpha_nj)
        - math.log(2.0 * degree)
        - math.log(beta_nj)
        - exponent * math.log(shifted_norm)
        - 0.5 * math.log(shots)
    ) / exponent
    try:
        delta_t = math.exp(log_delta)
    except OverflowError as exc:
        raise ValueError("optimized_delta_t is not representable") from exc
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("optimized_delta_t must be positive and finite")

    return np.array([energy_shift, shifted_norm, delta_t, alpha_nj, beta_nj], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
coefficients = np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60], dtype=float)
""",
            "call": "compute_msd_preprocessing(coefficients, 4, 3, 1.0e14, -1.2, 1.8)",
            "gold_call": "_oracle_compute_msd_preprocessing(coefficients, 4, 3, 1.0e14, -1.2, 1.8)",
        },
        {
            "setup": """import numpy as np
coefficients = np.array([-0.5, 0.0, 0.5], dtype=float)
""",
            "call": "compute_msd_preprocessing(coefficients, 1, 1, 1.0, -2.0, 2.0)",
            "gold_call": "_oracle_compute_msd_preprocessing(coefficients, 1, 1, 1.0, -2.0, 2.0)",
        },
        {
            "setup": """import numpy as np
coefficients = np.array([-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60], dtype=float)

def pack_extreme_result(result):
    values = np.asarray(result, dtype=np.float64)
    if values.shape != (5,):
        raise ValueError("preprocessing must have shape (5,)")
    delta_t = float(values[2])
    valid_delta = bool(np.isfinite(delta_t) and delta_t > 0.0)
    log_delta = float(np.log(delta_t)) if valid_delta else 0.0
    return np.concatenate([
        values, np.asarray([float(valid_delta), log_delta], dtype=np.float64),
    ])
""",
            "call": "pack_extreme_result(compute_msd_preprocessing(coefficients, 4, 3, 1.0e200, -1.0e40, 1.0e40))",
            "gold_call": "pack_extreme_result(_oracle_compute_msd_preprocessing(coefficients, 4, 3, 1.0e200, -1.0e40, 1.0e40))",
        },
        {
            "setup": """import numpy as np
coefficients = np.array([-0.5, 0.0, 0.5], dtype=float)
def run_model():
    try:
        compute_msd_preprocessing(coefficients, 2, 1, 0.0, -1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_msd_preprocessing(coefficients, 2, 1, 0.0, -1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
coefficients = np.array([-0.5, 0.0, 0.5], dtype=float)
def run_model():
    try:
        compute_msd_preprocessing(coefficients, 2, 1, 100.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_msd_preprocessing(coefficients, 2, 1, 100.0, 1.0, 1.0)
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

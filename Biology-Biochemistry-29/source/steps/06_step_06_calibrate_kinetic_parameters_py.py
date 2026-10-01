"""
Solve the coupled inverse problem using the constant Yp, two fraction observations, and one logarithmic-sensitivity observation. Enumerate algebraic inverse candidates, reconstruct rates, enforce the parameter box, and validate candidates using the original forward model.

The two fraction observations provide s(v) and r(v). At the sensitivity experiment, use the original forward and implicit-sensitivity equations to eliminate the nuisance equilibrium concentration and obtain a scalar inverse condition. Reconstruct the four rates from each real candidate and reject singular, nonpositive, out-of-bounds, or inconsistent solutions. A local optimizer alone does not certify inverse uniqueness. The supplied instance is intended to have one admissible candidate in the specified parameter box; raise ValueError when the supplied data do not yield exactly one distinct admissible solution.

Returns
-------
np.ndarray of shape (4,) within the supplied bounds, or ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def calibrate_kinetic_parameters(
    fixed_rates: np.ndarray,
    observations: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Return [k3,k6,k8,k12] from the coupled calibration problem.

    fixed_rates is a positive length-14 vector with placeholders
    at indices 2,5,7,11. observations=[Yp,F_A,F_B,G_C].
    bounds is a finite positive (4,2) array of lower/upper bounds.

    Use the fixed calibration conditions specified in Step 4.
    Enumerate inverse candidates, reconstruct the original rates,
    enforce the bounds, and validate the original forward model.
    Require exactly one distinct admissible solution in the box,
    with maximum absolute original-observation residual <=1e-8.
    Numerical refinement is permitted, but the solution must not
    be hard-coded.

    Returns:
        np.ndarray: Four finite rates [k3,k6,k8,k12] within
            the supplied parameter bounds.

    Raises:
        ValueError: If the inputs or parameter bounds are invalid,
            no admissible calibration satisfies the residual
            tolerance, or more than one distinct admissible
            inverse solution survives in the parameter box.
    """
    return np.empty(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial import Polynomial

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _assemble_rates(base, unknown):
    k = np.asarray(base, dtype=float).copy()
    k[[2, 5, 7, 11]] = unknown
    return k

def _predict_calibration(k):
    c = _oracle_compute_equilibrium_coefficients(k)
    return np.array([
        c[6],
        _oracle_compute_fraction_sensitivity(c, 1.0, 0.8)[0],
        _oracle_compute_fraction_sensitivity(c, 1.8, 0.3)[0],
        _oracle_compute_fraction_sensitivity(c, 1.4, 1.1)[1]
    ])

def _inverse_polynomial(observations, relation):
    yp, f1, f2, g_signed = observations
    p0, p1, q0, q1 = relation

    v = Polynomial([0.0, 1.0])
    den = Polynomial([q0, q1])
    sn = Polynomial([p0, p1])

    x1 = 1.0 - f1
    y1 = 1.8 - yp
    rn = (den / x1 - sn) * (y1 - v * x1)

    tc, yc = 1.4, 2.5 - yp
    g = -g_signed

    a = tc * v * sn
    b = -(yc * sn + tc * v * den + rn)

    poly = (
        yc * (rn + 2*g*a)**2 * den
        - g*b*b*(rn + g*a)
    )

    if poly.degree() > 12 or not np.any(poly.coef):
        raise ValueError("invalid inverse polynomial")

    return poly

def _oracle_calibrate_kinetic_parameters(
    fixed_rates, observations, bounds
):
    import numpy as np
    from numpy.polynomial import Polynomial

    _context = {
        "np": np,
        "Polynomial": Polynomial,
        "_checked_array": _checked_array,
        "_assemble_rates": _assemble_rates,
        "_predict_calibration": _predict_calibration,
        "_inverse_polynomial": _inverse_polynomial,
        "_oracle_compute_equilibrium_coefficients":
            _oracle_compute_equilibrium_coefficients,
        "_oracle_solve_positive_equilibrium":
            _oracle_solve_positive_equilibrium,
        "_oracle_compute_fraction_sensitivity":
            _oracle_compute_fraction_sensitivity,
        "_oracle_construct_inverse_constraints":
            _oracle_construct_inverse_constraints,
        "_oracle_reconstruct_kinetic_parameters":
            _oracle_reconstruct_kinetic_parameters
    }

    for fn in (
        _oracle_compute_equilibrium_coefficients,
        _oracle_solve_positive_equilibrium,
        _oracle_compute_fraction_sensitivity,
        _oracle_construct_inverse_constraints,
        _oracle_reconstruct_kinetic_parameters,
        _assemble_rates,
        _predict_calibration,
        _inverse_polynomial
    ):
        fn.__globals__.update(_context)

    k = _checked_array(fixed_rates, (14,), "fixed_rates", True)
    o = _checked_array(observations, (4,), "observations")
    b = _checked_array(bounds, (4, 2), "bounds", True)

    if np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("invalid bounds")

    relation = _oracle_construct_inverse_constraints(o)
    polynomial = _inverse_polynomial(o, relation)
    candidates = []

    for root in polynomial.roots():
        if abs(root.imag) > 1e-7:
            continue

        v = float(root.real)
        if v <= 0:
            continue

        p0, p1, q0, q1 = relation
        den = q0 + q1*v

        if abs(den) < 1e-12:
            continue

        s = (p0 + p1*v)/den
        x1 = 1.0 - o[1]
        y1 = 1.8 - o[0]
        r = (1.0/x1 - s)*(y1 - v*x1)

        if (
            not np.all(np.isfinite([s, v, r]))
            or s <= 1
            or r <= 0
            or not 0 < s-v < 1
        ):
            continue

        try:
            unknown = _oracle_reconstruct_kinetic_parameters(
                [s, v, r], o[0], k
            )

            if (
                np.any(unknown < b[:, 0] - 1e-8)
                or np.any(unknown > b[:, 1] + 1e-8)
            ):
                continue

            unknown = np.clip(unknown, b[:, 0], b[:, 1])

            residual = (
                _predict_calibration(_assemble_rates(k, unknown))
                - o
            )

            if np.max(np.abs(residual)) > 1e-8:
                continue

            if not any(
                np.allclose(
                    unknown, old, rtol=1e-5, atol=1e-8
                )
                for old in candidates
            ):
                candidates.append(unknown)

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError,
            FloatingPointError
        ):
            continue

    if len(candidates) != 1:
        raise ValueError("expected exactly one admissible inverse solution")

    return candidates[0]

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
            "description": "Calibrate all four rates from the supplied observations.",
            "setup": common,
            "call": "calibrate_kinetic_parameters(base,observations,bounds)",
            "gold_call": "_oracle_calibrate_kinetic_parameters(base,observations,bounds)",
            "tol": 1e-9
        },
        {
            "description": "Calibrate an independent synthetic parameter vector.",
            "setup": common,
            "call": "calibrate_kinetic_parameters(base,alt_observations,bounds)",
            "gold_call": "_oracle_calibrate_kinetic_parameters(base,alt_observations,bounds)",
            "tol": 1e-9
        },
        {
            "description": "A malformed parameter box is rejected.",
            "setup": common + """
bad = bounds.copy()
bad[0] = [1.5,0.7]
""",
            "call": "capture(calibrate_kinetic_parameters,base,observations,bad)",
            "gold_call": "capture(_oracle_calibrate_kinetic_parameters,base,observations,bad)",
            "tol": 1e-9
        }
    ]

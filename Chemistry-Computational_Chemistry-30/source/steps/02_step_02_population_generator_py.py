"""
Construct and return the linear population generator in the order S₁, T₂, T₁, with z=α exp(−Δ/θ) and d=κ+k_nr,T1. Include both T₁ escape pathways. Its column sums must reproduce the total-population balance (S+U+L)′=−S/τ_S1−dL.

Set all annihilation coefficients to zero. The radiative rate κ=1 s⁻¹ is a hypothetical independently known input. Internal transfers must cancel from the summed population balance.

Returns
-------
np.ndarray, shape (3,3), float64; column-vector generator in state order [S1,T2,T1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def population_generator(tau_s: float, i: float, u: float, c: float,
                         z: float, d: float) -> np.ndarray:
    """Build M for x'=Mx, x=(S1,T2,T1), without annihilation.

    tau_s is intrinsic singlet lifetime (s); i,u,c are positive ISC,
    upper-triplet RISC and IC rates (s^-1). z>=0 is the dimensionless
    shared factor alpha*exp(-gap/theta). T1->S1 has rate u*z,
    T1->T2 has rate c*z, and d>=0 is total T1 ground-state loss.
    S1 ground-state loss is 1/tau_s. No other transitions occur.

    Returns
    -------
    np.ndarray
        Shape (3,3), finite float column-vector generator. Its column
        sums encode ground-state loss and internal transfer conserves
        total excitation population.

    Raises
    ------
    ValueError
        If an input is not a finite real scalar, tau_s,i,u,c are not
        positive, z or d is negative, or matrix entries overflow.
    """
    return np.empty((3, 3), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_population_generator(tau_s, i, u, c, z, d):
    import numpy as np
    try:
        raw = [tau_s, i, u, c, z, d]
        if np.iscomplexobj(raw):
            raise ValueError("real rates required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalar inputs required")
    if np.any(v[:4] <= 0) or np.any(v[4:] < 0):
        raise ValueError("invalid rates")
    tau_s, i, u, c, z, d = v
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        m = np.array([[-1.0/tau_s-i, u, u*z],
                      [i, -u-c, c*z],
                      [0.0, c, -d-(u+c)*z]], dtype=float)
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix overflow")
    return m

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "",
            "call": "population_generator(6e-9, 6e7, 7.5e5, 30., 5e-6, 4.0)",
            "gold_call": "_oracle_population_generator(6e-9, 6e7, 7.5e5, 30., 5e-6, 4.0)"
        },
        {
            "setup": "",
            "call": "population_generator(6e-9, 6e7, 7.5e5, 30., 0., 8.333333333333334)",
            "gold_call": "_oracle_population_generator(6e-9, 6e7, 7.5e5, 30., 0., 8.333333333333334)"
        },
        {
            "setup": "",
            "call": "population_generator(0.2, 2., 3., 0.5, 0.1, 0.)",
            "gold_call": "_oracle_population_generator(0.2, 2., 3., 0.5, 0.1, 0.)"
        },
        {
            "setup": "def _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(population_generator, 6e-9, 6e7, 7.5e5, 30., -0.1, 1.)",
            "gold_call": "_expect_value_error(_oracle_population_generator, 6e-9, 6e7, 7.5e5, 30., -0.1, 1.)"
        }
    ]

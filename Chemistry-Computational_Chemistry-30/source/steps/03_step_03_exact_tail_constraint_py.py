"""
Derive the relation between d and z required by the target slow decay rate λ. Eliminate the singlet and upper-triplet amplitudes at the target eigenvalue and evaluate the resulting coefficients.

A measured decay rate is the negative of the corresponding eigenvalue of the coupled generator, not simply the sum of rates leaving T₁. Retain λ in the elimination rather than assuming instantaneous equilibration.

Returns
-------
np.ndarray, shape (3,), float64: [a (s^-1), B (s^-2), H (s^-1)] for d=lambda-H*z
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tail_coefficients(tau_s: float, i: float, u: float, c: float,
                      decay: float) -> np.ndarray:
    """Eliminate S1 and T2 at a specified exp(-decay*t) eigenmode.

    tau_s is positive intrinsic singlet lifetime in seconds; i,u,c are
    positive ISC, RISC2, and IC rates in s^-1; decay>0 is the measured
    inverse tail lifetime, not an intrinsic loss. In the model of step 2
    write a=1/tau_s+i, b=u+c, B=(a-decay)*(b-decay)-i*u.
    Determine the coefficient H such that the exact eigenvalue equation
    for the T1 loss becomes d=decay-H*z. Do not set decay to zero in
    the fast-state elimination.

    Returns
    -------
    np.ndarray
        Shape (3,), [a, B, H] in s^-1, s^-2, s^-1.

    Raises
    ------
    ValueError
        If inputs are not finite real positive scalars, a<=decay,
        b<=decay, B<=0, H<=0, or any computed coefficient is nonfinite.
        This contract restricts the target to below the fast-block poles.
    """
    return np.empty(3, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_tail_coefficients(tau_s, i, u, c, decay):
    import numpy as np
    try:
        raw = [tau_s, i, u, c, decay]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (5,) or not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("finite positive inputs required")
    tau_s, i, u, c, decay = v
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = 1.0/tau_s+i
        b = u+c
        B = (a-decay)*(b-decay)-i*u
    if not np.all(np.isfinite([a,b,B])) or a <= decay or b <= decay or B <= 0:
        raise ValueError("target must lie below fast-block poles")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        H = b-c*((a-decay)*c+i*u)/B
    if not np.isfinite(H) or H <= 0:
        raise ValueError("finite positive H required")
    return np.array([a, B, H], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "",
            "call": "tail_coefficients(6e-9, 6e7, 7.5e5, 30., 1.0/0.120)",
            "gold_call": "_oracle_tail_coefficients(6e-9, 6e7, 7.5e5, 30., 1.0/0.120)"
        },
        {
            "setup": "",
            "call": "tail_coefficients(0.1, 2., 3., 0.5, 0.01)",
            "gold_call": "_oracle_tail_coefficients(0.1, 2., 3., 0.5, 0.01)"
        },
        {
            "setup": "",
            "call": "tail_coefficients(6e-9, 6e7, 7.5e5, 1e-6, 1.0)",
            "gold_call": "_oracle_tail_coefficients(6e-9, 6e7, 7.5e5, 1e-6, 1.0)"
        },
        {
            "setup": "def _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(tail_coefficients, 0.1, 2., 3., 0.5, 3.5)",
            "gold_call": "_expect_value_error(_oracle_tail_coefficients, 0.1, 2., 3., 0.5, 3.5)"
        }
    ]

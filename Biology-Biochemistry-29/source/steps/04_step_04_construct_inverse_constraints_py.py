"""
Eliminate the experiment-specific equilibrium concentrations from the two calibration fraction observations. Return a rational relation between the shared normalized inverse variables s and v.

Define s=S/B, v=C/B, r=beta/B, and x=BQ/T1=1-F. The equilibrium equation is (1-sx)(y0-vT1x)=rx. For experiments A=(1.0,0.8) and B=(1.8,0.3), let x1=1-F_A, x2=1-F_B, y1=1.8-Yp, and y2=2.1-Yp. Eliminate the common r between the two experiments to derive a rational relation s(v). The sensitivity observation is retained for the subsequent coupled inverse solve. Do not assume the calibration equations are separable.

Returns
-------
np.ndarray of shape (4,), containing finite rational-relation coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_inverse_constraints(
    observations: np.ndarray,
) -> np.ndarray:
    """Return [p0,p1,q0,q1], defining s(v)=(p0+p1*v)/(q0+q1*v).

    observations=[Yp,F_A,F_B,G_C], where A uses totals
    (1.0,0.8), B uses (1.8,0.3), and C uses (1.4,1.1).
    Eliminate the shared inverse variable using the two fraction
    observations. Require finite observations, 0<Yp<1.8,
    0<F_A,F_B<1, and G_C<0.

    Returns:
        np.ndarray: Four finite rational-relation coefficients.

    Raises:
        ValueError: If the observation vector is malformed,
            nonfinite, or violates the stated admissibility
            conditions.
    """
    return np.empty(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _oracle_construct_inverse_constraints(observations):
    import numpy as np

    o = _checked_array(observations, (4,), "observations")
    yp, f1, f2, g = o

    if (
        yp <= 0 or yp >= 1.8
        or not (0 < f1 < 1 and 0 < f2 < 1)
        or g >= 0
    ):
        raise ValueError("invalid calibration observations")

    x1, x2 = 1-f1, 1-f2
    y1, y2 = 1.8-yp, 2.1-yp

    return _checked_array(
        np.array([
            -(y1/x1-y2/x2),
            -0.8,
            -(y1-y2),
            x1-1.8*x2
        ]),
        (4,),
        "inverse relation"
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
"""

    return [
        {
            "description": "Normal inverse relation.",
            "setup": common,
            "call": "construct_inverse_constraints(observations)",
            "gold_call": "_oracle_construct_inverse_constraints(observations)",
            "tol": 1e-9
        },
        {
            "description": "Independent synthetic inverse relation.",
            "setup": common,
            "call": "construct_inverse_constraints(alt_observations)",
            "gold_call": "_oracle_construct_inverse_constraints(alt_observations)",
            "tol": 1e-9
        },
        {
            "description": "An impossible measured Yp is rejected.",
            "setup": common + """
bad = observations.copy()
bad[0] = 1.8
""",
            "call": "capture(construct_inverse_constraints, bad)",
            "gold_call": "capture(_oracle_construct_inverse_constraints, bad)",
            "tol": 1e-9
        }
    ]

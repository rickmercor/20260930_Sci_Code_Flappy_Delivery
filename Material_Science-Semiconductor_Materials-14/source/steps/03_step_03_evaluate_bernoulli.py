"""
Evaluate the Bernoulli function that carries the exponential fitting of the harmonic-average flux, in a form that stays accurate at the removable singularity and is finite in both asymptotic limits.

Write the one-dimensional steady continuity equation for electrons along a mesh edge, assume the potential varies linearly between the two endpoints and the current density is constant on the edge, and the resulting two-point boundary value problem can be integrated exactly. The solution weights the two endpoint densities not by one half each, as a centred difference would, but by the function B(t) = t / (exp(t) - 1) evaluated at the potential difference across the edge measured in units of the thermal voltage. This is the Bernoulli function, and its two limits are why exponentially fitted schemes are stable on convection-dominated transport: B(t) tends to zero as t grows, so the flux forgets the downwind node entirely in the drift-dominated regime, while B(0) = 1 recovers the plain arithmetic difference in the diffusion-dominated regime.




The same function reappears when the harmonic average of the exponential of a linear potential profile is computed along an edge, which is the route by which it enters the discrete duality construction. Averaging exp of minus the linear projection of the potential over an edge and inverting the result produces exp of the potential at one endpoint multiplied by B of the potential difference between the two endpoints, so the harmonic average and the exponential fitting are the same object seen from two directions.




Evaluating B naively is unsafe in exactly the places the drift-diffusion problem visits, and four branches are needed to cover them. At t equal to zero the expression is zero divided by zero, and for small nonzero t the subtraction exp(t) - 1 loses most of its significant digits to cancellation, so the quotient is dominated by rounding error long before the argument underflows; the remedy is the Taylor expansion B(t) = 1 - t/2 + t**2/12 + O(t**4), whose neglected term is of fourth order and therefore below rounding for arguments smaller than about 1e-8. For moderate arguments the expm1 primitive computes exp(t) - 1 without the cancellation. At the other end, a PN junction at equilibrium already carries a potential difference of roughly twenty-three thermal voltages across the depletion region, and a Newton iteration passing through a poor intermediate state can present arguments of several hundred. For large positive t the naive quotient divides a finite number by an overflowing one, so the algebraically equivalent form t * exp(-t) is used instead, which underflows gracefully to zero, and for large negative t the exponential vanishes and B(t) approaches -t, which is the exact asymptote.

Returns
-------
np.ndarray of the same shape as the input, float: the Bernoulli function B(t) = t / (exp(t) - 1) with the removable singularity at the origin filled in as B(0) = 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def evaluate_bernoulli(t: np.ndarray) -> np.ndarray:
    """Evaluate the Bernoulli function B(t) = t / (exp(t) - 1), with B(0) = 1.

    Parameters
    ----------
    t : np.ndarray
        Array of scaled potential differences, of any shape. Values may span
        the full double precision range in both directions.

    Returns
    -------
    values : np.ndarray
        Array of the same shape as t holding B(t), finite and strictly
        positive for every finite input.

    Raises
    ------
    ValueError
        If t contains a non-finite value.
    """
    return values  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_bernoulli(t: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    t = np.asarray(t, dtype=float)

    if not np.all(np.isfinite(t)):
        raise ValueError(
            "t must contain only finite values"
        )

    values = np.empty(
        t.shape,
        dtype=float,
    )

    small = np.abs(t) < 1.0e-8
    large_pos = t > 500.0
    large_neg = t < -500.0
    middle = ~(
        small
        | large_pos
        | large_neg
    )

    ts = t[small]
    values[small] = (
        1.0
        - 0.5 * ts
        + ts * ts / 12.0
    )

    values[large_pos] = (
        t[large_pos]
        * np.exp(-t[large_pos])
    )

    values[large_neg] = -t[large_neg]

    tm = t[middle]
    values[middle] = (
        tm / np.expm1(tm)
    )

    return values

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: potential differences of the size a biased PN junction produces ---
        {
            "setup": """import numpy as np
t = np.linspace(-25.0, 25.0, 101)
""",
            "call": "float(1.0e12 * (_a := np.ravel(evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: the removable singularity and the cancellation region around it ---
        {
            "setup": """import numpy as np
t = np.array([0.0, 1e-16, -1e-16, 1e-9, -1e-9, 1e-8, -1e-8, 1e-7, -1e-7,
              1e-6, -1e-6, 1e-4, -1e-4])
""",
            "call": "float(1.0e12 * (_a := np.ravel(evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: arguments large enough to overflow the naive quotient ---
        {
            "setup": """import numpy as np
t = np.array([500.0, 500.5, 501.0, 710.0, 1.0e4, -500.0, -500.5, -710.0, -1.0e4])
""",
            "call": "float(1.0e12 * (_a := np.ravel(evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: two-dimensional input, so the shape must be preserved ---
        {
            "setup": """import numpy as np
t = np.array([[-3.0, 0.0, 3.0], [-40.0, 1e-12, 40.0]])
""",
            "call": "float(1.0e12 * (_a := np.ravel(evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_evaluate_bernoulli(t)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: a non-finite entry ---
        {
            "setup": """import numpy as np
t = np.array([0.0, np.inf, 1.0])
def run_model():
    try:
        evaluate_bernoulli(t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_bernoulli(t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a NaN entry ---
        {
            "setup": """import numpy as np
t = np.array([np.nan])
def run_model():
    try:
        evaluate_bernoulli(t)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_bernoulli(t)
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

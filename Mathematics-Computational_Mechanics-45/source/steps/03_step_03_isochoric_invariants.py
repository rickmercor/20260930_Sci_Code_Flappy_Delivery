"""
Return the two isochoric invariants of a symmetric positive definite strain

measure from its three principal values, in the polyconvex form used by

spline-based constitutive functions.

Isotropic constitutive functions at finite strain depend on a strain measure only

through its principal invariants, and incompressibility is imposed by taking the

Flory split first, so the invariants are formed from the unimodular part of the

tensor. For a symmetric positive definite tensor with principal values b_A the

unimodular part has principal values b_A * det(b)^(-1/3), and its first invariant

is I1 = sum_A b_A. The second invariant is not used in its raw form. Spline-based

and polynomial-type polyconvex energies use the transformed measure I2 =

w^(3/2) - 3 * sqrt(3), where w = tr(cof) is the sum of the pairwise products of

the unimodular principal values. The shift and the exponent are what make the

measure vanish in the undeformed state, where every unimodular principal value is

one and w = 3, so that both invariants start at their normalisation points, I1 = 3

and I2 = 0. This is the convention the interpolation domains of the two free

energy contributions are anchored to.

Returns
-------
tuple of two native Python floats, the first and the transformed second isochoric invariant
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def isochoric_invariants(principal_values):
    """Return the two isochoric invariants of a strain measure.

    Parameters
    ----------
    principal_values : array-like of shape (3,)
        Principal values of a symmetric positive definite strain measure.

    Returns
    -------
    invariants : tuple of two floats
        The first isochoric invariant and the transformed second isochoric
        invariant, the latter zero in the undeformed state.

    Raises
    ------
    ValueError
        If principal_values does not hold exactly three entries or if any entry
        is not strictly positive.
    """
    return invariants

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_SQRT3 = np.sqrt(3.0)


def _oracle_isochoric_invariants(principal_values):
    """Reference implementation."""
    b = np.asarray(principal_values, dtype=float)
    if b.shape != (3,):
        raise ValueError("principal_values must hold exactly three entries")
    if not np.all(b > 0.0):
        raise ValueError("principal values must be strictly positive")

    determinant = float(b[0] * b[1] * b[2])
    unimodular = b * determinant ** (-1.0 / 3.0)
    I1 = float(unimodular.sum())
    w = float(unimodular[0] * unimodular[1] + unimodular[0] * unimodular[2]
              + unimodular[1] * unimodular[2])
    I2 = float(w ** 1.5 - 3.0 * _SQRT3)
    return I1, I2

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# Elastic left Cauchy-Green principal values after the first increment of the
# prompt configuration, written out as a literal so this file runs on its own.
_ELASTIC_PRINCIPALS = np.array([1.089210527987, 0.958173343964, 0.958173343964])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario ---
        {
            "setup": """principal_values = _ELASTIC_PRINCIPALS.copy()
""",
            "call": "[round(v, 12) for v in isochoric_invariants(principal_values)]",
            "gold_call": "[round(v, 12) for v in _oracle_isochoric_invariants(principal_values)]",
        },
        # --- Boundary case: the undeformed state, where both invariants sit at
        # their normalisation points, and a state that is not unimodular ---
        {
            "setup": """principal_values = np.array([1.0, 1.0, 1.0])
scaled = np.array([8.0, 8.0, 8.0])
""",
            "call": "[[round(v, 12) for v in isochoric_invariants(b)] for b in (principal_values, scaled)]",
            "gold_call": "[[round(v, 12) for v in _oracle_isochoric_invariants(b)] for b in (principal_values, scaled)]",
        },
        # --- Edge case: a non-positive principal value must raise ValueError ---
        {
            "setup": """principal_values = np.array([1.2, 0.0, 0.8])

def run_model():
    try:
        isochoric_invariants(principal_values)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_isochoric_invariants(principal_values)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

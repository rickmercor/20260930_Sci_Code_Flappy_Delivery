"""
Return the trial elastic state of one increment of incompressible homogeneous

uniaxial loading, together with the squared principal stretches that map between

the elastic strain measure and the internal variable.

Isotropic constitutive functions at finite strain depend on a strain measure only

through its principal invariants, and incompressibility is imposed by taking the

Flory split first, so the invariants are formed from the unimodular part of the

tensor. For a symmetric positive definite tensor with principal values b_A the

unimodular part has principal values b_A * det(b)^(-1/3), and its first invariant

is the sum of those. The second invariant is not used in its raw form.

Spline-based and polynomial-type polyconvex energies replace it by a transformed

measure formed from the second principal invariant of the unimodular tensor,

raised to a fixed power and shifted, so that the result stays polyconvex and

vanishes in the undeformed state. That normalisation is what lets each invariant

start at the left endpoint of the interpolation domain it is anchored to, the

first at the value it takes when the tensor is the identity and the second at

zero.

Returns
-------
tuple of two np.ndarray of shape (3,), the trial elastic principal values and the squared principal stretches, both as float64 arrays
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def uniaxial_kinematics(stretch: float, cv_inverse):
    """Return the trial elastic state of one uniaxial increment.

    Parameters
    ----------
    stretch : float
        End-of-increment stretch in the loading direction.
    cv_inverse : array-like of shape (3,)
        Principal values of the inverse viscous right Cauchy-Green tensor
        carried over from the previous increment.

    Returns
    -------
    trial_state : tuple of two np.ndarray of shape (3,)
        The principal values of the trial elastic left Cauchy-Green tensor and
        the squared principal stretches of the deformation gradient.

    Raises
    ------
    ValueError
        If stretch is not strictly positive or if cv_inverse does not hold
        exactly three strictly positive entries.
    """
    return trial_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_uniaxial_kinematics(stretch: float, cv_inverse):
    """Reference implementation."""
    internal = np.asarray(cv_inverse, dtype=float)
    if internal.shape != (3,):
        raise ValueError("cv_inverse must hold exactly three entries")
    if not np.all(internal > 0.0):
        raise ValueError("cv_inverse entries must be strictly positive")
    if not float(stretch) > 0.0:
        raise ValueError("stretch must be strictly positive")

    lam = float(stretch)
    stretch_squared = np.array([lam ** 2, 1.0 / lam, 1.0 / lam])
    beta_trial = stretch_squared * internal
    return beta_trial, stretch_squared

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# Internal variable after the first increment of the prompt configuration,
# written out as a literal so this file runs on its own.
_CV_INVERSE = np.array([0.987946057131, 1.006082011162, 1.006082011162])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the second increment of the prompt history ---
        {
            "setup": """stretch = 1.10
cv_inverse = _CV_INVERSE.copy()
""",
            "call": "[[round(v, 12) for v in a.tolist()] for a in uniaxial_kinematics(stretch, cv_inverse)]",
            "gold_call": "[[round(v, 12) for v in a.tolist()] for a in _oracle_uniaxial_kinematics(stretch, cv_inverse)]",
        },
        # --- Boundary case: the first increment, started from a virgin internal
        # state, and the undeformed configuration ---
        {
            "setup": """cv_inverse = np.ones(3)
""",
            "call": "[[[round(v, 12) for v in a.tolist()] for a in uniaxial_kinematics(s, cv_inverse)] for s in (1.05, 1.0)]",
            "gold_call": "[[[round(v, 12) for v in a.tolist()] for a in _oracle_uniaxial_kinematics(s, cv_inverse)] for s in (1.05, 1.0)]",
        },
        # --- Edge case: a non-positive stretch must raise ValueError ---
        {
            "setup": """cv_inverse = np.ones(3)

def run_model():
    try:
        uniaxial_kinematics(0.0, cv_inverse)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_uniaxial_kinematics(0.0, cv_inverse)
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

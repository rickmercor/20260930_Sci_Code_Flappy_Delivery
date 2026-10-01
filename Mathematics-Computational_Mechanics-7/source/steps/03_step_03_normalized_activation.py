"""
Evaluate the normalized smooth activation candidate and its first two equivalent-strain derivatives for every element.

Define the yield strain, origin slope, and normalized transition coordinate by



$$

\\epsilon_y=\\dfrac{\\sigma_y}{3\\mu},

\\qquad

s_0=\\dfrac{1}{1+\\exp\\beta},

\\qquad

z=\\beta(\\dfrac{q}{\\epsilon_y}-1).

$$



With the softplus function $\\mathcal S(z)=\\log(1+\\exp z)$, the instantaneous candidate is



$$

\\hat h=\\dfrac{\\epsilon_y}{\\beta(1-s_0)}

[

\\mathcal S(z)-\\mathcal S(-\\beta)-s_0\\beta\\dfrac{q}{\\epsilon_y}

].

$$



Its derivative with respect to equivalent strain is



$$

\\hat h'=\\dfrac{(1+\\exp(-z))^{-1}-s_0}{1-s_0}.

$$



The returned curvature is the exact derivative of this normalized sigmoid with respect to $q$; it is not a finite-difference estimate. The constant subtraction enforces $\\hat h(0)=0$, while the linear subtraction also enforces $\\hat h'(0)=0$. At large strain the derivative approaches one and the curvature approaches zero. The candidate is an instantaneous constitutive quantity and does not by itself create irreversibility; the maximum-history projection is applied in the next step. Stable softplus and sigmoid evaluations are required to avoid overflow at large $|z|$ without suppressing the finite near-transition curvature.

Returns
-------
Return a tuple containing the dimensionless activation candidate and its first and second derivatives as arrays of shape (n_elements,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def normalized_activation(
    equivalent_strain: np.ndarray,
    shear_modulus: np.ndarray,
    yield_stress: np.ndarray,
    sharpness: np.ndarray,
) -> tuple:
    r"""Evaluate a normalized smooth candidate and its first two derivatives.

    Inputs must be finite, nonempty one-dimensional arrays of identical shape with q >=
    0 and all material parameters positive. Invalid inputs raise ValueError.

    Parameters
    ----------
    equivalent_strain : np.ndarray
        Nonnegative dimensionless q, shape (n_elements,).
    shear_modulus : np.ndarray
        Positive shear modulus in MPa, same shape.
    yield_stress : np.ndarray
        Positive yield scale in MPa, same shape; zero yield is outside the domain.
    sharpness : np.ndarray
        Positive dimensionless transition sharpness, same shape.

    Raises
    ------
    ValueError
        If an input is nonfinite; arrays are empty, not one-dimensional, or have
        different shapes; equivalent strain is negative; a material parameter is
        nonpositive; or the computed activation exceeds finite arithmetic range.

    Returns
    -------
    tuple
        (candidate, derivative, curvature): dimensionless float arrays of shape
        (n_elements,), where the derivatives are with respect to equivalent strain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _oracle_normalized_activation(
    equivalent_strain: np.ndarray,
    shear_modulus: np.ndarray,
    yield_stress: np.ndarray,
    sharpness: np.ndarray,
) -> tuple:
    q, mu, sy, beta = _matched_vectors(
        equivalent_strain, shear_modulus, yield_stress, sharpness
    )
    if np.any(q < 0) or np.any(mu <= 0) or np.any(sy <= 0) or np.any(beta <= 0):
        raise ValueError("invalid strain or activation parameters")
    threshold = sy / (3.0 * mu)
    s0 = np.exp(-np.logaddexp(0.0, beta))
    z = beta * (q / threshold - 1.0)
    candidate = (
        threshold
        / (beta * (1.0 - s0))
        * (np.logaddexp(0.0, z) - np.logaddexp(0.0, -beta) - s0 * beta * q / threshold)
    )
    sigmoid = np.exp(-np.logaddexp(0.0, -z))
    derivative = (sigmoid - s0) / (1.0 - s0)
    curvature = beta * sigmoid * (1.0 - sigmoid) / (threshold * (1.0 - s0))
    candidate = np.maximum(candidate, 0.0)
    derivative = np.clip(derivative, 0.0, 1.0)
    candidate[q == 0.0] = 0.0
    derivative[q == 0.0] = 0.0
    if not all(
        np.all(np.isfinite(value)) for value in (candidate, derivative, curvature)
    ):
        raise ValueError("activation exceeds finite arithmetic range")
    return candidate, derivative, curvature

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
q = np.array([.03, .0866666666667, .18])
mu = np.full(3, 20 / 2.6)
sy = np.full(3, 2.)
beta = np.full(3, 12.)
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in normalized_activation(q, mu, sy, beta)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_normalized_activation(q, mu, sy, beta)])",
        },
        {
            "setup": """
import numpy as np
q = np.zeros(2)
mu = np.array([5., 12.])
sy = np.array([1., 1.2])
beta = np.array([4., 8.])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in normalized_activation(q, mu, sy, beta)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_normalized_activation(q, mu, sy, beta)])",
        },
        {
            "setup": """
import numpy as np
q = np.array([.039999, .040001, 10.])
mu = np.full(3, 10.)
sy = np.full(3, 1.2)
beta = np.full(3, 1000.)
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in normalized_activation(q, mu, sy, beta)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_normalized_activation(q, mu, sy, beta)])",
        },
        {
            "setup": """
import numpy as np
q = np.array([.1])
mu = np.array([10.])
sy = np.array([0.])
beta = np.array([12.])
def _invalid_status(function):
    try:
        function(q, mu, sy, beta)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(normalized_activation)",
            "gold_call": "_invalid_status(_oracle_normalized_activation)",
        },
    ]

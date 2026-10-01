"""
Compute the inelastic work, attenuation factor, and first two derivatives of their product with respect to history.

The history-dependent inelastic work is



$$

W_p(h)=\\sigma_yh+\\dfrac{1}{2}Hh^2.

$$



The survival factor associated with inelastic attenuation is



$$

L(h)=\\exp(-Ch)=1-D.

$$



Only $W_p$ is multiplied by $L$; the volumetric and deviatoric elastic energies are not attenuated. Differentiating the product gives



$$

G=\\dfrac{d[L(h)W_p(h)]}{dh}

=L(\\sigma_y+Hh-CW_p).

$$



The returned second derivative is obtained by differentiating $G$ with respect to $h$, including every product-rule contribution from $L$. Omitting an occurrence of the attenuation derivative produces a symmetric and finite tangent with the wrong loading stiffness. When $C=0$, $L=1$, $G=\\sigma_y+Hh$, and $G'=H$. For sufficiently large history, either derivative may be negative and must not be clipped. Because the attenuation acts only on inelastic work, the frozen elastic unloading modulus is unchanged.

Returns
-------
Return a tuple containing the inelastic work, survival factor, and first two derivatives of attenuated work as arrays of shape (n_elements,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def inelastic_work(
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    attenuation_rate: np.ndarray,
) -> tuple:
    r"""Differentiate the attenuated inelastic-work contribution twice with respect to history.

    Inputs must be finite, nonempty matched vectors with h >= 0, yield_stress > 0, H >=
    0, C >= 0. Invalid inputs raise ValueError.

    Parameters
    ----------
    history : np.ndarray
        Nonnegative dimensionless h, shape (n_elements,).
    yield_stress : np.ndarray
        Positive yield scale in MPa, same shape.
    hardening : np.ndarray
        Nonnegative hardening scale H in MPa, same shape.
    attenuation_rate : np.ndarray
        Nonnegative dimensionless C, same shape.

    Raises
    ------
    ValueError
        If inputs are nonfinite; arrays are empty, not one-dimensional, or have
        different shapes; history, hardening, or attenuation rate is negative; yield
        stress is nonpositive; or a computed result is nonfinite.

    Returns
    -------
    tuple
        (work, survival, derivative, second_derivative): arrays of shape
        (n_elements,), respectively MPa, dimensionless, MPa, and MPa.
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


def _oracle_inelastic_work(
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    attenuation_rate: np.ndarray,
) -> tuple:
    h, sy, hardening, rate = _matched_vectors(
        history, yield_stress, hardening, attenuation_rate
    )
    if np.any(h < 0) or np.any(sy <= 0) or np.any(hardening < 0) or np.any(rate < 0):
        raise ValueError("invalid inelastic-work parameters")
    work = sy * h + 0.5 * hardening * h**2
    survival = np.exp(-rate * h)
    derivative = survival * (sy + hardening * h - rate * work)
    second_derivative = survival * (
        hardening - 2.0 * rate * (sy + hardening * h) + rate**2 * work
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (work, survival, derivative, second_derivative)
    ):
        raise ValueError("inelastic work exceeds finite arithmetic range")
    return work, survival, derivative, second_derivative

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
h = np.array([0., .05, .14])
sy = np.array([2., 2., 1.2])
H = np.array([.5, .5, 2.4])
C = np.array([2.2, 2.2, 4.4])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in inelastic_work(h, sy, H, C)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_inelastic_work(h, sy, H, C)])",
        },
        {
            "setup": """
import numpy as np
h = np.array([0., .05, .14])
sy = np.array([2., 2., 1.2])
H = np.array([.5, .5, 2.4])
C = np.array([2.2, 2.2, 4.4])
C[:] = 0
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in inelastic_work(h, sy, H, C)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_inelastic_work(h, sy, H, C)])",
        },
        {
            "setup": """
import numpy as np
h = np.array([0., .05, .14])
sy = np.array([2., 2., 1.2])
H = np.array([.5, .5, 2.4])
C = np.array([2.2, 2.2, 4.4])
H[:] = 0
h[-1] = 1.
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in inelastic_work(h, sy, H, C)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_inelastic_work(h, sy, H, C)])",
        },
        {
            "setup": """
import numpy as np
h = np.array([0., .05, .14])
sy = np.array([2., 2., 1.2])
H = np.array([.5, .5, 2.4])
C = np.array([2.2, 2.2, 4.4])
C[0] = -1
def _invalid_status(function):
    try:
        function(h, sy, H, C)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(inelastic_work)",
            "gold_call": "_invalid_status(_oracle_inelastic_work)",
        },
    ]

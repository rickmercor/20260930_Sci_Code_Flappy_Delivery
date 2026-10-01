"""
Compute the propagated radial field and its directional derivative.

Write $A=M_{11}$, $B=M_{12}$, $C=M_{21}$, and $D=M_{22}$ for the position-slope transfer entries. The supplied response $\dot M$ is a directional derivative of this same matrix; both arrays have shape $(2,2)$ and satisfy the determinant-one convention.



$$AD-BC=1.$$



The exit field retains the radial structure of the specified entrance Bessel-Gaussian beam. Use a positive-sign quadratic diffraction phase, with the position-independent propagation phase set to zero, and differentiate only the supplied transfer direction while holding $k,w,\theta,r$ fixed. The field is defined continuously at $B=0$.

Returns
-------
A complex array of shape $(2,N)$ contains the field and its directional derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_field_response(
    radii: "np.ndarray",
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
) -> "np.ndarray":
    r"""Return the radial field and its transfer-direction response.

    Parameters
    ----------
    radii : np.ndarray
        Nonempty finite nonnegative real vector of radii, in $\mathrm{m}$.
    matrix : np.ndarray
        Real finite shape $(2,2)$ position-slope transfer with
        $|\det M-1|\leq10^{-8}$.
    response : np.ndarray
        Real finite shape $(2,2)$ derivative $\dot M$ satisfying
        $|D\dot A+A\dot D-C\dot B-B\dot C|\leq10^{-7}$.
    wave_number : float
        Positive finite internal $k$, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite input amplitude-envelope radius $w$, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle $0\leq\theta<\pi/2$, in radians.

    Returns
    -------
    field : np.ndarray
        Complex shape $(2,N)$ array: row zero is $U(r)$ and row one is
        $\dot U(r)$ at the input radii in their original order.

    Raises
    ------
    ValueError
        If radii or matrix shapes, reality, finiteness, determinant, or
        tangent constraints fail; if a beam parameter is invalid; or if
        the evaluated field or response is nonfinite.

    Notes
    -----
    Inputs are not modified. The stated phase convention also applies at $B=0$.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import jve


def _real_array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or entries")
    return array


def _oracle_compute_field_response(
    radii: "np.ndarray",
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
) -> "np.ndarray":
    if np.iscomplexobj(radii):
        raise ValueError("radii must be real")
    radii = np.asarray(radii, dtype=float)
    if (
        radii.ndim != 1
        or radii.size == 0
        or not np.isfinite(radii).all()
        or np.any(radii < 0.0)
    ):
        raise ValueError("radii must be a nonempty nonnegative finite vector")
    matrix = _real_array(matrix, (2, 2), "matrix")
    response = _real_array(response, (2, 2), "response")
    a, b, c, d = matrix.ravel()
    da, db, dc, dd = response.ravel()
    if abs(a * d - b * c - 1.0) > 1e-8:
        raise ValueError("matrix must have determinant one")
    if abs(d * da + a * dd - c * db - b * dc) > 1e-7:
        raise ValueError("response must preserve the determinant to first order")
    wave_number, waist, transverse = _beam_parameters(wave_number, waist, cone_angle)
    q = a + 2j * b / (wave_number * waist**2)
    dq = da + 2j * db / (wave_number * waist**2)
    h = d / waist**2 - 0.5j * wave_number * c
    dh = dd / waist**2 - 0.5j * wave_number * dc
    argument = transverse * radii / q
    argument_response = -argument * dq / q
    exponent = -h * radii**2 / q - 0.5j * b * transverse**2 / (wave_number * q)
    log_response = (
        -dq / q
        - radii**2 * (dh / q - h * dq / q**2)
        - 0.5j * transverse**2 / wave_number * (db / q - b * dq / q**2)
    )
    # Both orders share the same real scaling, so no logarithmic Bessel ratio is needed.
    prefactor = np.exp(exponent + np.abs(argument.imag)) / q
    zero = jve(0, argument)
    first = jve(1, argument)
    field = prefactor * zero
    derivative = prefactor * (zero * log_response - first * argument_response)
    result = np.array([field, derivative])
    if not np.isfinite(result).all():
        raise ValueError("field evaluation is nonfinite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and input-preservation cases."""
    return [
        {
            "setup": """import numpy as np
r = np.array([0.0, 1e-5, 3e-5])
m = np.array([[.8,.004],[-90.,.8]])
dm = np.array([[.004,0.0],[.8,0.0]])
def _preserved(function, *arguments):
    copied = [argument.copy() if isinstance(argument, np.ndarray) else argument for argument in arguments]
    before = [argument.copy() if isinstance(argument, np.ndarray) else argument for argument in copied]
    result = function(*copied)
    for previous, current in zip(before, copied):
        if isinstance(previous, np.ndarray):
            assert np.array_equal(previous, current), "input array was modified"
    return result
""",
            "call": "_preserved(compute_field_response, r.copy(), m.copy(), dm.copy(), 1e7, 7e-5, "
            ".003)",
            "gold_call": "_preserved(_oracle_compute_field_response, r.copy(), m.copy(), dm.copy(), "
            "1e7, 7e-5, .003)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
r = np.array([0.0, 2e-5])
m = np.eye(2)
dm = np.zeros((2,2))""",
            "call": "compute_field_response(r.copy(), m.copy(), dm.copy(), 1e7, 7e-5, .003)",
            "gold_call": "_oracle_compute_field_response(r.copy(), m.copy(), dm.copy(), 1e7, 7e-5, "
            ".003)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
r = np.array([4e-5, 0.0, 2e-5])
m = np.array([[-2.,0.0],[10.,-.5]])
dm = np.array([[0.0,0.0],[1.0,0.0]])""",
            "call": "compute_field_response(r.copy(), m.copy(), dm.copy(), 1e7, 7e-5, .003)",
            "gold_call": "_oracle_compute_field_response(r.copy(), m.copy(), dm.copy(), 1e7, 7e-5, "
            ".003)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(np.array([-1.0]), np.eye(2), np.zeros((2,2)), 1e7, 7e-5, .003)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_field_response)",
            "gold_call": "_raises(_oracle_compute_field_response)",
        },
    ]

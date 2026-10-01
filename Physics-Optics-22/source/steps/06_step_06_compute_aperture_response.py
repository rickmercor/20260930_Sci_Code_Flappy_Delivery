"""
Integrate the captured optical power fraction and its directional derivative.

The detector is a circular disk of radius $R$. Its physical area measure is



$$dA=2\pi r\,dr.$$



The two returned values are the captured fraction and its directional response under a change of the supplied transfer. The unit-amplitude entrance power $P_0$ stays fixed because the entrance beam parameters do not vary in this direction. The aperture includes the radial rings of the propagated field.

Returns
-------
A real array of shape $(2,)$ contains the captured power fraction and its directional derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_aperture_response(
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
    aperture: float,
    order: int = 96,
) -> "np.ndarray":
    r"""Return the aperture fraction and its transfer-direction derivative.

    Parameters
    ----------
    matrix : np.ndarray
        Finite real determinant-one transfer of shape $(2,2)$, with
        determinant tolerance $10^{-8}$.
    response : np.ndarray
        Finite real shape $(2,2)$ tangent derivative, with absolute
        first-order determinant tolerance $10^{-7}$.
    wave_number : float
        Positive finite internal wave number, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite entrance envelope radius, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle in $[0,\pi/2)$, in radians.
    aperture : float
        Finite aperture radius $R\geq0$, in $\mathrm{m}$.
    order : int, optional
        Integer radial quadrature order at least $16$, default $96$.

    Returns
    -------
    capture : np.ndarray
        Shape $(2,)$ real array $(F,\dot F)$, where $F$ is dimensionless
        and $\dot F$ is per unit of the directional parameter.

    Raises
    ------
    ValueError
        If aperture or order is invalid, or a matrix, direction, or beam
        input violates the field-response contracts, including nonfinite
        evaluated fields.

    Notes
    -----
    Inputs are not modified. A zero aperture returns two zeros.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_compute_aperture_response(
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
    aperture: float,
    order: int = 96,
) -> "np.ndarray":
    aperture = _finite_scalar(aperture, "aperture")
    nodes, weights = leggauss(_quadrature_order(order))
    radii = aperture * (nodes + 1.0) / 2.0
    field, derivative = _oracle_compute_field_response(
        radii, matrix, response, wave_number, waist, cone_angle
    )
    power = _oracle_compute_input_power(wave_number, waist, cone_angle)
    measure = np.pi * aperture * weights * radii / power
    fraction = np.dot(measure, np.abs(field) ** 2)
    variation = np.dot(measure, 2.0 * np.real(np.conj(field) * derivative))
    return np.array([fraction, variation])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and input-preservation cases."""
    return [
        {
            "setup": """import numpy as np
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
            "call": "_preserved(compute_aperture_response, m.copy(), dm.copy(), 1e7, 7e-5, .003, "
            "2.5e-5)",
            "gold_call": "_preserved(_oracle_compute_aperture_response, m.copy(), dm.copy(), 1e7, "
            "7e-5, .003, 2.5e-5)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
m = np.eye(2)
dm = np.zeros((2,2))""",
            "call": "compute_aperture_response(m.copy(), dm.copy(), 1e7, 7e-5, .003, 0.0)",
            "gold_call": "_oracle_compute_aperture_response(m.copy(), dm.copy(), 1e7, 7e-5, .003, 0.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
m = np.eye(2)
dm = np.array([[0.0,.01],[0.0,0.0]])""",
            "call": "compute_aperture_response(m.copy(), dm.copy(), 1e7, 7e-5, 0.0, 1e-4)",
            "gold_call": "_oracle_compute_aperture_response(m.copy(), dm.copy(), 1e7, 7e-5, 0.0, 1e-4)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(np.eye(2), np.zeros((2,2)), 1e7, 7e-5, .003, -.01)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_aperture_response)",
            "gold_call": "_raises(_oracle_compute_aperture_response)",
        },
    ]

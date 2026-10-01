"""
Evaluate the signed pressure and its local differential from an interfacial fit.

The excess free energy per area is $2\sigma(h)$ for two interfaces. Thus, when

$\sigma=a\exp(bh)+c$ is in mN/m and $h$ is in angstroms,



$$

\Pi=-20ab\exp(bh)\quad\text{in MPa}.

$$



Return columns $\Pi$, $\partial_h\Pi$, $\partial_a\Pi$, $\partial_b\Pi$,

$\partial_c\Pi$ with the other arguments held fixed. The derivative units are,

respectively, MPa/angstrom, MPa/(mN/m), MPa angstrom, and MPa/(mN/m).

Returns
-------
Return a numerical array of shape (m, 5) containing pressure and its four ordered partial derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def differentiate_disjoining_pressure(
    thickness: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    r"""Evaluate pressure and partial derivatives.

    Parameters
    ----------
    thickness : np.ndarray
        Shape $(m,)$, $m\geq1$, positive finite thicknesses in angstroms.
    parameters : np.ndarray
        Shape $(3,)$, finite real $(a,b,c)$ with $b<0$.

    Returns
    -------
    result : np.ndarray
        Shape $(m,5)$, columns $(\Pi,\Pi_h,\Pi_a,\Pi_b,\Pi_c)$ with the
        pressure and derivative units defined in the scientific background.

    Raises
    ------
    ValueError
        If arrays have invalid shapes, nonreal or nonfinite entries, nonpositive
        thicknesses, or a nonnegative decay coefficient.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_differentiate_disjoining_pressure(
    thickness: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness = _finite_array(thickness, "thickness", 1)
    parameters = _finite_array(parameters, "parameters", 1)
    if thickness.size < 1 or np.any(thickness <= 0):
        raise ValueError("thicknesses must be positive and nonempty")
    if parameters.shape != (3,) or parameters[1] >= 0:
        raise ValueError("parameters must be (a, b, c) with b < 0")
    a, b, _ = parameters
    exponential = np.exp(b * thickness)
    pressure = -20.0 * a * b * exponential
    return np.column_stack(
        (
            pressure,
            b * pressure,
            -20.0 * b * exponential,
            -20.0 * a * exponential * (1.0 + b * thickness),
            np.zeros_like(thickness),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerical test specifications."""
    return [
        {
            "setup": """import numpy as np
thickness = np.array([3., 8., 20.])
parameters = np.array([-20., -.1, 40.])
""",
            "call": "differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
            "gold_call": "_oracle_differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
        },
        {
            "setup": """import numpy as np
thickness = np.array([10.])
parameters = np.array([0., -.1, 40.])
""",
            "call": "differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
            "gold_call": "_oracle_differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
        },
        {
            "setup": """import numpy as np
thickness = np.array([1000., 2.])
parameters = np.array([3., -.01, -2.])
""",
            "call": "differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
            "gold_call": "_oracle_differentiate_disjoining_pressure(thickness.copy(), parameters.copy())",
        },
        {
            "setup": """import numpy as np
thickness = np.array([1.])
parameters = np.array([1., 0., 2.])

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(differentiate_disjoining_pressure, thickness.copy(), parameters.copy())",
            "gold_call": "_capture_value_error(_oracle_differentiate_disjoining_pressure, thickness.copy(), parameters.copy())",
        },
    ]

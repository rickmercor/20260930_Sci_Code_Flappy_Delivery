"""
Find the global absolute extremum of a second-harmonic CP-phase error.

For $f(\delta)=h_0+h_1\cos\delta+h_2\sin\delta+h_3\cos2\delta+h_4\sin2\delta$,

find $\max_{0\leq\delta<2\pi}|f(\delta)|$. Interior extrema of a nonzero

absolute maximum satisfy $f'(\delta)=0$; all-zero coefficients return zero.

With $t=\tan(\delta/2)$, the stationarity polynomial, in descending powers, is



$$

(-h_2+2h_4)t^4+(-2h_1+8h_3)t^3-12h_4t^2

+(-2h_1-8h_3)t+(h_2+2h_4)=0.

$$



Evaluate all real-root phases together with $\delta=0$ and $\delta=\pi$,

the latter covering the point omitted by the half-angle coordinate.

A constant polynomial returns $|h_0|$; missing high-order terms reduce the

polynomial degree. The result is a continuous-phase maximum, not a grid maximum.

Returns
-------
A finite nonnegative Python float containing the global absolute error over one continuous phase period.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maximize_phase_error(coefficients: "np.ndarray") -> float:
    r"""Find the global absolute extremum of a second-harmonic CP-phase error.

    Parameters
    ----------
    coefficients : np.ndarray
        Finite real shape (5,) coefficients ordered as constant, cos(delta),
        sin(delta), cos(2 delta), sin(2 delta).

    Returns
    -------
    maximum : float
        Finite nonnegative maximum absolute value over a complete phase period.

    Raises
    ------
    ValueError
        If coefficient shape, reality or finiteness is invalid, or the
        requested finite maximum overflows floating-point representation.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _phase_extrema(coefficients):
    h0, a, b, c, d = coefficients
    polynomial = np.array(
        [-b + 2 * d, -2 * a + 8 * c, -12 * d, -2 * a - 8 * c, b + 2 * d]
    )
    scale = np.max(np.abs(polynomial))
    phases = [0.0, np.pi]
    if scale > 0.0:
        roots = np.roots(np.trim_zeros(polynomial / scale, "f"))
        for root in roots:
            if abs(root.imag) <= 1e-7 * (1.0 + abs(root.real)):
                phases.append(float((2.0 * np.arctan(root.real)) % (2.0 * np.pi)))
    phases = np.array(phases)
    values = (
        h0
        + a * np.cos(phases)
        + b * np.sin(phases)
        + c * np.cos(2 * phases)
        + d * np.sin(2 * phases)
    )
    return phases, values


def _oracle_maximize_phase_error(coefficients: "np.ndarray") -> float:
    data = _real_array(coefficients, (5,), "coefficients")
    scale = float(np.max(np.abs(data)))
    if scale == 0.0:
        return 0.0
    _, values = _phase_extrema(data / scale)
    result = float(np.max(np.abs(values)) * scale)
    if not np.isfinite(result):
        raise ValueError("maximum must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nh=np.array([.2,-.4,.7,.3,-.2])\n",
            "call": "maximize_phase_error(h.copy())",
            "gold_call": "_oracle_maximize_phase_error(h.copy())",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nh=np.zeros(5)\n",
            "call": "maximize_phase_error(h.copy())",
            "gold_call": "_oracle_maximize_phase_error(h.copy())",
            "tol": 0.0,
        },
        {
            "setup": "import numpy as np\nh=np.array([-2.,0.,0.,0.,0.])\n",
            "call": "maximize_phase_error(h.copy())",
            "gold_call": "_oracle_maximize_phase_error(h.copy())",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nh=np.array([0.,3.,4.,0.,0.])\n",
            "call": "maximize_phase_error(h.copy())",
            "gold_call": "_oracle_maximize_phase_error(h.copy())",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nh=np.array([0.,0.,0.,-1.,0.])\n",
            "call": "maximize_phase_error(h.copy())",
            "gold_call": "_oracle_maximize_phase_error(h.copy())",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nh=np.zeros(4)\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: maximize_phase_error(h.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_maximize_phase_error(h.copy()))",
            "tol": 0.0,
        },
    ]

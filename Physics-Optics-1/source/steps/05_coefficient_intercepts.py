"""
Convert the Fourier amplitudes of the approximant cell trace, obtained at reference height $h_0$, into the paper's real intercepts $b_n^{(q)}$ of the logarithmic coefficient magnitudes.

Inside a connected nonsingular strip the cell trace depends on the phase and on $h$ only through one complex combination, so each Fourier amplitude varies with $h$ in a rigidly prescribed, order-dependent way. The paper's intercept $b_n^{(q)}$ removes that known $h$-dependence and is therefore independent of the reference height at which the coefficient was sampled. Consult the paper's own derivation of this exact $h$-dependence for the scaling and for the intercept's definition.

Returns
-------
np.ndarray of shape (len(coeffs),), float: the intercepts $[b_0^{(q)},b_1^{(q)},\dots]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def coefficient_intercepts(coeffs: "np.ndarray", q: int, h0: float) -> "np.ndarray":
    r"""Intercepts of the logarithmic Fourier-coefficient magnitudes.

    Args:
        coeffs (np.ndarray): complex Fourier amplitudes $[C_0,C_1,\dots]$ of the cell
            trace, all sampled at the same reference height $h_0$; index = order $n$.
        q (int): approximant denominator $q$ used to build the coefficients.
        h0 (float): the reference continuation height $h_0$ at which they were sampled.

    Raises:
        ValueError: if any coefficient is exactly zero.

    Expected return:
        np.ndarray of float, same length as coeffs; for exact coefficients the result
        does not depend on $h_0$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: coefficient_intercepts
def _oracle_coefficient_intercepts(coeffs: "np.ndarray", q: int, h0: float) -> "np.ndarray":
    coeffs = np.asarray(coeffs, dtype=complex)
    if np.any(np.abs(coeffs) == 0.0):
        raise ValueError("zero Fourier coefficient has no logarithmic intercept")
    n = np.arange(len(coeffs))
    return np.log(np.abs(coeffs)) / q - n * h0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'v = coefficient_intercepts(np.array([2.0, 3.0e5, 1.0e9 + 2.0e8j]), 21, 0.8)'
            ),
            "call": 'np.round(v, 8)',
            "gold_call": 'np.round(_oracle_coefficient_intercepts(np.array([2.0, 3.0e5, 1.0e9 + 2.0e8j]), 21, 0.8), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = coefficient_intercepts(np.array([1.0 - 1.0j, -4.0e3, 7.5e7j, 2.0e10]), 34, 0.55)'
            ),
            "call": 'np.round(v, 8)',
            "gold_call": 'np.round(_oracle_coefficient_intercepts(np.array([1.0 - 1.0j, -4.0e3, 7.5e7j, 2.0e10]), 34, 0.55), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = coefficient_intercepts(np.array([0.3, 5.0, 80.0]), 5, 0.0)'
            ),
            "call": 'np.round(v, 8)',
            "gold_call": 'np.round(_oracle_coefficient_intercepts(np.array([0.3, 5.0, 80.0]), 5, 0.0), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = coefficient_intercepts(np.array(trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 64), dtype=complex), 34, 0.8)'
            ),
            "call": 'np.round(v, 7)',
            "gold_call": 'np.round(_oracle_coefficient_intercepts(np.array(_oracle_trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 64), dtype=complex), 34, 0.8), 7)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        coefficient_intercepts(np.array([1.0, 0.0, 2.0]), 13, 0.8)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_coefficient_intercepts(np.array([1.0, 0.0, 2.0]), 13, 0.8)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2'
            ),
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]

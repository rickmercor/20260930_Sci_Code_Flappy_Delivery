"""
Resolve the one-sided Fourier coefficients of a reciprocal material.

The constitutive wave equation contains the reciprocal of alpha as a spatial function. Constant-term dominance makes its one-sided Fourier series convergent. A finite constitutive polynomial generally has infinitely many reciprocal harmonics; retain enough of them to act on the requested diffraction orders, without a weak-contrast expansion.

Returns
-------
return inverse_coefficients: finite complex ndarray (order_count,) containing harmonics 0 through order_count-1 of the reciprocal spatial profile, without modifying coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reciprocal_fourier(coefficients, order_count):
    """Return coefficients 0,...,order_count-1 of 1/a(y).

    coefficients is a nonempty finite numeric one-dimensional array for
    a(y)=sum(coefficients[h]*exp(1j*h*K*y)), h>=0. Require
    abs(coefficients[0])>sum(abs(coefficients[1:])) so the reciprocal has
    a convergent one-sided expansion. order_count is a positive non-Boolean
    integer. Return a finite complex array of shape (order_count,) without
    changing the input. Raise ValueError for invalid inputs or nonfinite
    output. Reciprocal coefficients beyond the material's polynomial
    degree generally do not vanish.
    """
    return np.zeros(order_count, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reciprocal_fourier(coefficients, order_count):
    import numpy as np

    raw = np.asarray(coefficients)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iufc" or not np.all(np.isfinite(raw)):
        raise ValueError("Expected a nonempty finite numeric coefficient vector.")
    values = raw.astype(complex)
    if abs(values[0]) <= np.sum(np.abs(values[1:])):
        raise ValueError("The constant coefficient must strictly dominate the tail.")
    if isinstance(order_count, (bool, np.bool_)) or not isinstance(order_count, (int, np.integer)) or order_count < 1:
        raise ValueError("order_count must be a positive integer.")
    inverse = np.zeros(order_count, dtype=complex)
    inverse[0] = 1 / values[0]
    for degree in range(1, order_count):
        inverse[degree] = -sum(
            values[harmonic] * inverse[degree - harmonic]
            for harmonic in range(1, min(degree + 1, len(values)))
        ) / values[0]
    if not np.all(np.isfinite(inverse)):
        raise ValueError("The reciprocal coefficients are not finite.")
    return inverse

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '',
      'call': 'reciprocal_fourier([2.0], 5)',
      'gold_call': '_oracle_reciprocal_fourier([2.0], 5)'},
     {'setup': '',
      'call': 'reciprocal_fourier([1.0, 0.2j], 6)',
      'gold_call': '_oracle_reciprocal_fourier([1.0, 0.2j], 6)'},
     {'setup': 'coefficients = np.array([2.4+0.45j, 0.35, 0.04j])',
      'call': 'reciprocal_fourier(coefficients, 8)',
      'gold_call': '_oracle_reciprocal_fourier(coefficients, 8)'},
     {'setup': 'def reciprocal_residual(fn):\n'
               '    coefficients = np.array([1.3 + 0.2j, 0.14 - 0.03j, -0.025j])\n'
               '    original = coefficients.copy()\n'
               '    inverse = fn(coefficients, 7)\n'
               '    residual = np.convolve(coefficients, inverse)[:7]\n'
               '    residual[0] -= 1\n'
               '    return max(float(np.max(np.abs(residual))), float(np.max(np.abs(coefficients - '
               'original))))\n',
      'call': 'reciprocal_residual(reciprocal_fourier)',
      'gold_call': 'reciprocal_residual(_oracle_reciprocal_fourier)'},
     {'setup': 'def rejects_zero_profile(fn):\n'
               '    try:\n'
               '        fn([0.0, 0.1], 3)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_zero_profile(reciprocal_fourier)',
      'gold_call': 'rejects_zero_profile(_oracle_reciprocal_fourier)'}]

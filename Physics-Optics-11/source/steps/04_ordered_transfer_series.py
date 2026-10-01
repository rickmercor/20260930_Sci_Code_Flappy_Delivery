"""
Compose an ordered layer stack through second order in total thickness.

Physical state evolution through successive layers is a product of matrix exponentials with the outgoing layer on the left. Its degree-two coefficient contains both the self-layer factorial terms and ordered cross-layer products. Thickness fractions remain attached to their generators, including under reversal or splitting of a layer.

Returns
-------
return state_series: complex ndarray (3,d,d) whose indices 0,1,2 multiply eta**0, eta**1, eta**2; the constant coefficient is identity, and d is the generator dimension.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ordered_transfer_series(generators, fractions):
    """Return the state-transfer coefficients [T0,T1,T2].

    generators has finite numeric shape (layers, dimension, dimension).
    fractions is a finite real nonnegative vector with one entry per layer,
    summing to one within absolute tolerance 1e-12. Inputs are ordered from
    the incident face to the exit face. The state transfer is the ordered
    product exp(1j*eta*f_last*G_last) ... exp(1j*eta*f_first*G_first).
    Return a complex array of shape (3, dimension, dimension) whose entries
    multiply eta**0, eta**1, eta**2, respectively; these are polynomial
    coefficients, not unscaled derivatives. T0 is identity. Zero-thickness
    layers are allowed. Raise ValueError for invalid inputs and preserve
    all inputs. Do not commute generators or normalize supplied fractions.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros(
        (
            3,
            np.asarray(generators).shape[-1],
            np.asarray(generators).shape[-1],
        ),
        dtype=complex,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ordered_transfer_series(generators, fractions):
    import numpy as np

    generators = np.asarray(generators)
    fractions = np.asarray(fractions)
    if (
        generators.ndim != 3
        or generators.shape[0] == 0
        or generators.shape[1] == 0
        or generators.shape[1] != generators.shape[2]
    ):
        raise ValueError("Expected a nonempty stack of square generators.")
    if generators.dtype.kind not in "iufc" or not np.all(
        np.isfinite(generators)
    ):
        raise ValueError("Generators must be finite numeric matrices.")
    if (
        fractions.shape != (generators.shape[0],)
        or fractions.dtype.kind not in "iuf"
        or not np.all(np.isfinite(fractions))
    ):
        raise ValueError(
            "Fractions must be a finite real vector, one per layer."
        )
    if np.any(fractions < 0) or abs(float(np.sum(fractions)) - 1) > 1e-12:
        raise ValueError("Nonnegative fractions must sum to one.")
    dimension = generators.shape[1]
    coefficients = np.zeros((3, dimension, dimension), dtype=complex)
    coefficients[0] = np.eye(dimension)
    for generator, fraction in zip(generators, fractions):
        first = 1j * fraction * generator
        coefficients[2] += first @ coefficients[1] + 0.5 * first @ first
        coefficients[1] += first
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("State coefficients are not finite.")
    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\ngenerator = np.diag([1.0, -0.4j])\n',
      'call': 'ordered_transfer_series([generator], [1.0])',
      'gold_call': '_oracle_ordered_transfer_series([generator], [1.0])'},
     {'setup': '\n'
               'first = np.array([[0.0, 1.0], [0.0, 0.0]])\n'
               'second = np.array([[0.0, 0.0], [2.0, 0.0]])\n'
               'expected = np.array(\n'
               '    [np.eye(2), 1j * (0.3 * first + 0.7 * second), -0.21 * second @ first]\n'
               ')\n',
      'call': 'ordered_transfer_series([first, second], [0.3, 0.7])',
      'gold_call': '_oracle_ordered_transfer_series([first, second], [0.3, 0.7])'},
     {'setup': '\ngenerator = np.array([[0.2j, 1.4], [0.8, -0.1j]])\n',
      'call': 'ordered_transfer_series([generator, generator, generator], [0.0, 0.25, 0.75])',
      'gold_call': '_oracle_ordered_transfer_series([generator, generator, generator], [0.0, 0.25, '
                   '0.75])'},
     {'setup': 'def rejects_bad_fractions(fn):\n'
               '    try:\n'
               '        fn([np.eye(2), np.eye(2)], [0.3, 0.6])\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_bad_fractions(ordered_transfer_series)',
      'gold_call': 'rejects_bad_fractions(_oracle_ordered_transfer_series)'},
     {'setup': 'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn([1e+200 * np.eye(2)], [1.0])\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(ordered_transfer_series)',
      'gold_call': 'rejects_nonfinite(_oracle_ordered_transfer_series)'}]

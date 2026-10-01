"""
Convert state-transfer coefficients into global plane-wave coordinates.

Vacuum right- and left-going waves have state columns [I;C] and [I;-C]. Face amplitudes and globally referenced amplitudes differ by their coordinate phase. Phase removal must be expanded consistently with state evolution before the specified truncated amplitude is formed; a vacuum layer must give the identity global transfer.

Returns
-------
return global_series: complex ndarray (3,2*J,2*J), mapping global [A_left,B_left] to [A_right,B_right] through eta**2, with constant coefficient identity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def global_transfer_series(state_series, cosines):
    """Return [M0,M1,M2] for globally referenced wave amplitudes.

    state_series has finite numeric shape (3,2*J,2*J), with its constant
    term equal to identity within absolute tolerance 1e-12. cosines is a
    finite vector of length J with each entry positive real or positive
    purely imaginary, and modulus greater than 1e-12. The state at a plane
    with coordinate x is the sum of [I;C] A exp(1j*k*C*x) and
    [I;-C] B exp(-1j*k*C*x), with C=diag(cosines). Both exterior media
    are vacuum, the entry face is x=0, and the exit face is k*x=eta.
    Derive M mapping [A_left,B_left] to [A_right,B_right] through eta**2.
    Coefficients are not derivatives. Remove the exit-face vacuum phase
    consistently for both propagation directions. Preserve inputs and
    raise ValueError for invalid shape, constant term, or branches.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros((3, 2 * len(cosines), 2 * len(cosines)), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_global_transfer_series(state_series, cosines):
    import numpy as np

    cosines = np.asarray(cosines)
    if (
        cosines.ndim != 1
        or cosines.size == 0
        or cosines.dtype.kind not in "iufc"
        or not np.all(np.isfinite(cosines))
    ):
        raise ValueError("Expected a finite numeric cosine vector.")
    cosines = cosines.astype(complex)
    branches = ((cosines.real > 0) & (cosines.imag == 0)) | (
        (cosines.real == 0) & (cosines.imag > 0)
    )
    if not np.all(branches) or np.any(np.abs(cosines) <= 1e-12):
        raise ValueError("Cosines must use nongrazing outgoing branches.")
    count = len(cosines)
    state_series = np.asarray(state_series)
    if (
        state_series.shape != (3, 2 * count, 2 * count)
        or state_series.dtype.kind not in "iufc"
        or not np.all(np.isfinite(state_series))
    ):
        raise ValueError("State coefficients have an invalid shape or values.")
    if not np.allclose(state_series[0], np.eye(2 * count), rtol=0, atol=1e-12):
        raise ValueError(
            "The constant state-transfer coefficient must be identity."
        )
    identity = np.eye(count)
    longitudinal = np.diag(cosines)
    basis = np.block([[identity, identity], [longitudinal, -longitudinal]])
    face_first = np.linalg.solve(basis, state_series[1] @ basis)
    face_second = np.linalg.solve(basis, state_series[2] @ basis)
    phase_first = np.diag(np.concatenate((-1j * cosines, 1j * cosines)))
    result = np.array(
        [
            np.eye(2 * count),
            phase_first + face_first,
            0.5 * phase_first @ phase_first
            + phase_first @ face_first
            + face_second,
        ]
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Global coefficients are not finite.")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\n'
               'cosines = np.array([0.8, 0.6, 1.2j])\n'
               'generator = np.block(\n'
               '    [[np.zeros((3, 3)), np.eye(3)], [np.diag(cosines**2), np.zeros((3, 3))]]\n'
               ')\n'
               'state_series = np.array(\n'
               '    [np.eye(6), 1j * generator, -0.5 * generator @ generator]\n'
               ')\n'
               'expected = np.zeros((3, 6, 6), dtype=complex)\n'
               'expected[0] = np.eye(6)\n',
      'call': 'global_transfer_series(state_series, cosines)',
      'gold_call': '_oracle_global_transfer_series(state_series, cosines)'},
     {'setup': '\n'
               'state_series = np.zeros((3, 2, 2), dtype=complex)\n'
               'state_series[0] = np.eye(2)\n'
               'expected = np.array([np.eye(2), np.diag([-0.8j, 0.8j]), -0.32 * np.eye(2)])\n',
      'call': 'global_transfer_series(state_series, [0.8])',
      'gold_call': '_oracle_global_transfer_series(state_series, [0.8])'},
     {'setup': '\n'
               'rng = np.random.default_rng(1105)\n'
               'state_series = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))\n'
               'state_series[0] = np.eye(4)\n'
               'cosines = np.array([0.9, 0.7j])\n',
      'call': 'global_transfer_series(state_series, cosines)',
      'gold_call': '_oracle_global_transfer_series(state_series, cosines)'},
     {'setup': 'def rejects_grazing_basis(fn):\n'
               '    series = np.array([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))])\n'
               '    try:\n'
               '        fn(series, [0.0])\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_grazing_basis(global_transfer_series)',
      'gold_call': 'rejects_grazing_basis(_oracle_global_transfer_series)'},
     {'setup': 'series = np.zeros((3, 2, 2), dtype=complex)\n'
               'series[0] = np.eye(2)\n'
               '\n'
               'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn(series, [1e+200])\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(global_transfer_series)',
      'gold_call': 'rejects_nonfinite(_oracle_global_transfer_series)'}]

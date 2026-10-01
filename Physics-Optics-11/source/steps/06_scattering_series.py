"""
Apply outgoing boundary conditions to a second-order transfer series.

A transfer matrix is not itself a transmission matrix: reflection adjusts the boundary state so that the incoming amplitude on the right vanishes. Solving this boundary condition coefficient by coefficient produces reflection and transmission series, including second-order feedback from first-order blocks.

Returns
-------
return np.zeros((2, 3, np.asarray(global_series).shape[-1] // 2), dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scattering_series(global_series):
    """Return reflection/transmission amplitude coefficients, shape (2,3,J).

    global_series has finite numeric shape (3,2*J,2*J), J>=1, and constant
    coefficient identity within absolute tolerance 1e-12. It maps
    [A_left,B_left] to [A_right,B_right] in global plane-wave coordinates.
    Incidence is from the left in order zero, A_left=e_0; there is no
    incidence from the right, B_right=0. Solve these boundary conditions
    as a series through eta**2. The first output index is reflected=0 or
    transmitted=1, the second is polynomial degree 0,1,2, and the third
    is diffraction order. Return coefficients, not amplitudes evaluated
    at a chosen thickness. Preserve inputs and raise ValueError for
    invalid values, shapes, or constant coefficient.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros(
        (2, 3, np.asarray(global_series).shape[-1] // 2), dtype=complex
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_scattering_series(global_series):
    import numpy as np

    global_series = np.asarray(global_series)
    if (
        global_series.ndim != 3
        or global_series.shape[0] != 3
        or global_series.shape[1] == 0
        or global_series.shape[1] != global_series.shape[2]
        or global_series.shape[1] % 2
    ):
        raise ValueError(
            "Expected three even-dimensional square coefficients."
        )
    if global_series.dtype.kind not in "iufc" or not np.all(
        np.isfinite(global_series)
    ):
        raise ValueError(
            "Transfer coefficients must be finite numeric values."
        )
    count = global_series.shape[1] // 2
    if not np.allclose(
        global_series[0], np.eye(2 * count), rtol=0, atol=1e-12
    ):
        raise ValueError("The constant transfer coefficient must be identity.")
    global_series = global_series.astype(np.complex128)
    first = global_series[1]
    second = global_series[2]
    incident = np.eye(count)[:, 0]
    result = np.zeros((2, 3, count), dtype=complex)
    result[1, 0] = incident
    result[0, 1] = -first[count:, :count] @ incident
    result[0, 2] = (
        first[count:, count:] @ first[count:, :count] - second[count:, :count]
    ) @ incident
    result[1, 1] = first[:count, :count] @ incident
    result[1, 2] = (
        second[:count, :count] - first[:count, count:] @ first[count:, :count]
    ) @ incident
    if not np.all(np.isfinite(result)):
        raise ValueError("Amplitude coefficients are not finite.")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\n'
               'transfer = np.zeros((3, 6, 6), dtype=complex)\n'
               'transfer[0] = np.eye(6)\n'
               'expected = np.zeros((2, 3, 3), dtype=complex)\n'
               'expected[1, 0, 0] = 1\n',
      'call': 'scattering_series(transfer)',
      'gold_call': '_oracle_scattering_series(transfer)'},
     {'setup': '\n'
               'transfer = np.array(\n'
               '    [np.eye(2), [[0, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=complex\n'
               ')\n',
      'call': 'scattering_series(transfer)',
      'gold_call': '_oracle_scattering_series(transfer)'},
     {'setup': '\n'
               'rng = np.random.default_rng(1106)\n'
               'transfer = rng.normal(size=(3, 6, 6)) + 1j * rng.normal(size=(3, 6, 6))\n'
               'transfer[0] = np.eye(6)\n',
      'call': 'scattering_series(transfer)',
      'gold_call': '_oracle_scattering_series(transfer)'},
     {'setup': 'def boundary_residual(fn):\n'
               '    rng = np.random.default_rng(7)\n'
               '    transfer = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))\n'
               '    transfer[0] = np.eye(4)\n'
               '    amplitudes = fn(transfer)\n'
               '    incident = np.array([1.0, 0.0])\n'
               '    outgoing = np.zeros((3, 4), dtype=complex)\n'
               '    for degree in range(3):\n'
               '        for coefficient in range(degree + 1):\n'
               '            incoming = np.r_[incident if degree - coefficient == 0 else np.zeros(2), '
               'amplitudes[0, degree - coefficient]]\n'
               '            outgoing[degree] += transfer[coefficient] @ incoming\n'
               '    expected = np.column_stack((amplitudes[1], np.zeros((3, 2))))\n'
               '    return float(np.max(np.abs(outgoing - expected)))\n',
      'call': 'boundary_residual(scattering_series)',
      'gold_call': 'boundary_residual(_oracle_scattering_series)'},
     {'setup': '\n'
               'transfer = np.zeros((3, 2, 2), dtype=np.int64)\n'
               'transfer[0] = np.eye(2, dtype=np.int64)\n'
               'transfer[1] = [[0, 10**10], [10**10, 0]]\n'
               'expected = np.array([[[0], [-1e10], [0]], [[1], [0], [-1e20]]], dtype=complex)\n',
      'call': 'scattering_series(transfer)',
      'gold_call': '_oracle_scattering_series(transfer)'},
     {'setup': 'transfer = np.zeros((3, 2, 2), dtype=complex)\n'
               'transfer[0] = np.eye(2)\n'
               'transfer[1] = 1e+200\n'
               '\n'
               'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn(transfer)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(scattering_series)',
      'gold_call': 'rejects_nonfinite(_oracle_scattering_series)'}]

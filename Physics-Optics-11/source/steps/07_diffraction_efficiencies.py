"""
Evaluate the specified amplitude polynomial and its vacuum flux weights.

Vacuum normal Poynting flux is proportional to the real longitudinal wavenumber times squared field amplitude. For this task, evaluate the amplitude polynomial before taking its modulus squared. Evanescent fields carry no outgoing normal power; the passive exact problem does not justify clipping or renormalizing a finite-order approximation.

Returns
-------
return efficiencies: finite real ndarray (2,J), reflection then transmission, with normal-flux weighting, zero evanescent power, and no further series truncation or renormalization.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def diffraction_efficiencies(amplitude_series, cosines, thickness):
    """Return reflected/transmitted power efficiencies, shape (2,J).

    amplitude_series has finite numeric shape (2,3,J), with polynomial
    coefficients through second order. thickness=eta=k*total_length is
    finite, real, nonnegative, and not Boolean. Evaluate the quadratic
    amplitude first, then take its squared modulus; do not Taylor-truncate
    the resulting power polynomial. The incident and exit media are vacuum
    with unit-amplitude incidence in order zero. Weight each propagating
    order by its normal vacuum Poynting flux relative to order zero.
    cosines is a finite vector: c_0 is positive real, and every entry is
    positive real or positive purely imaginary, of modulus above 1e-12.
    Evanescent orders carry zero normal far-field power. Do not renormalize
    the efficiencies to sum to one or clip absorption/gain effects.
    Preserve inputs and raise ValueError for invalid inputs/nonfinite output.
    """
    return np.zeros((2, len(cosines)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_diffraction_efficiencies(amplitude_series, cosines, thickness):
    import numpy as np

    raw_thickness = np.asarray(thickness)
    if (
        raw_thickness.ndim != 0
        or raw_thickness.dtype.kind not in "iuf"
        or not np.isfinite(raw_thickness)
        or raw_thickness < 0
    ):
        raise ValueError("thickness must be a finite nonnegative real scalar.")
    thickness = float(raw_thickness)
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
    if (
        not np.all(branches)
        or np.any(np.abs(cosines) <= 1e-12)
        or cosines[0].real <= 0
        or cosines[0].imag != 0
    ):
        raise ValueError(
            "Expected outgoing branches and a propagating incident order."
        )
    amplitude_series = np.asarray(amplitude_series)
    if (
        amplitude_series.shape != (2, 3, len(cosines))
        or amplitude_series.dtype.kind not in "iufc"
        or not np.all(np.isfinite(amplitude_series))
    ):
        raise ValueError(
            "Amplitude coefficients have invalid shape or values."
        )
    amplitude_series = amplitude_series.astype(np.complex128)
    amplitudes = (
        amplitude_series[:, 2] * thickness + amplitude_series[:, 1]
    ) * thickness + amplitude_series[:, 0]
    powers = np.abs(amplitudes) ** 2 * (cosines.real / cosines[0].real)
    if not np.all(np.isfinite(powers)):
        raise ValueError("The resulting efficiencies are not finite.")
    return powers

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\nseries = np.zeros((2, 3, 1), dtype=complex)\nseries[1, 0, 0] = 1\n',
      'call': 'diffraction_efficiencies(series, [0.8], 0.0)',
      'gold_call': '_oracle_diffraction_efficiencies(series, [0.8], 0.0)'},
     {'setup': '\n'
               'series = np.zeros((2, 3, 3), dtype=complex)\n'
               'series[0, 0] = [0.2, 0.3j, 5.0]\n'
               'series[1, 0] = [0.8, 0.4, 7j]\n',
      'call': 'diffraction_efficiencies(series, [0.8, 0.6, 1.2j], 0.2)',
      'gold_call': '_oracle_diffraction_efficiencies(series, [0.8, 0.6, 1.2j], 0.2)'},
     {'setup': '\n'
               'series = np.zeros((2, 3, 2), dtype=complex)\n'
               'series[1, 1, 1] = 1.0\n'
               'series[1, 2, 1] = 3.0j\n',
      'call': 'diffraction_efficiencies(series, [0.8, 0.4], 0.2)',
      'gold_call': '_oracle_diffraction_efficiencies(series, [0.8, 0.4], 0.2)'},
     {'setup': 'def rejects_negative_thickness(fn):\n'
               '    try:\n'
               '        fn(np.zeros((2, 3, 1)), [1.0], -0.1)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_negative_thickness(diffraction_efficiencies)',
      'gold_call': 'rejects_negative_thickness(_oracle_diffraction_efficiencies)'},
     {'setup': '',
      'call': 'diffraction_efficiencies(np.zeros((2, 3, 1)), [1.0], 1e200)',
      'gold_call': '_oracle_diffraction_efficiencies(np.zeros((2, 3, 1)), [1.0], 1e+200)'},
     {'setup': 'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn(np.ones((2, 3, 1)), [1.0], 1e+200)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(diffraction_efficiencies)',
      'gold_call': 'rejects_nonfinite(_oracle_diffraction_efficiencies)'}]

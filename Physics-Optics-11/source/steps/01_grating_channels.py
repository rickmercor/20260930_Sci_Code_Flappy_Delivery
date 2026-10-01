"""
Construct the causal diffraction orders and outgoing vacuum branches.

The transverse Fourier wavevector is conserved modulo positive grating harmonics. For left-incident order zero, use increasing nonnegative diffraction indices and the outgoing square-root branch. An imaginary longitudinal cosine represents an evanescent field, not a propagating angle; exact and near grazing channels are excluded by the declared tolerance.

Returns
-------
return sines, cosines: real ndarray (J,) and complex ndarray (J,), both finite, with J=order_count and the outgoing branch; input arrays are unchanged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def grating_channels(incident_sine, reciprocal_ratio, order_count):
    """Return (sines, cosines), each with shape (order_count,).

    Orders are j=0,...,order_count-1, with s_j=incident_sine+j*K/k
    and reciprocal_ratio=K/k. The incident sine is finite, real, and
    strictly between -1 and 1; reciprocal_ratio is finite and positive.
    order_count is a positive integer, not Boolean. The outgoing branch
    c_j=sqrt(1-s_j**2) is positive real for propagating orders and positive
    imaginary for evanescent orders. Raise ValueError for invalid inputs,
    nonfinite generated values, or abs(1-s_j**2)<=1e-12 (grazing).
    Do not reorder channels or replace an evanescent cosine by a real one.
    """
    return np.zeros(order_count), np.zeros(order_count, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_grating_channels(incident_sine, reciprocal_ratio, order_count):
    import numpy as np

    parameters = []
    for value in (incident_sine, reciprocal_ratio):
        raw = np.asarray(value)
        if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
            raise ValueError("Channel parameters must be finite real scalars.")
        parameters.append(float(raw))
    incident_sine, reciprocal_ratio = parameters
    if not -1 < incident_sine < 1 or reciprocal_ratio <= 0:
        raise ValueError("The incident channel and reciprocal ratio are invalid.")
    if isinstance(order_count, (bool, np.bool_)) or not isinstance(order_count, (int, np.integer)) or order_count < 1:
        raise ValueError("order_count must be a positive integer.")
    sines = incident_sine + reciprocal_ratio * np.arange(order_count)
    radicands = 1 - sines**2
    if not np.all(np.isfinite(radicands)) or np.any(np.abs(radicands) <= 1e-12):
        raise ValueError("Nonfinite or grazing diffraction channels are unsupported.")
    return sines, np.sqrt(radicands.astype(complex))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '',
      'call': 'np.stack(grating_channels(-0.6, 0.39, 5))',
      'gold_call': 'np.stack(_oracle_grating_channels(-0.6, 0.39, 5))'},
     {'setup': '',
      'call': 'np.stack(grating_channels(0.0, 0.4, 4))',
      'gold_call': 'np.stack(_oracle_grating_channels(0.0, 0.4, 4))'},
     {'setup': '',
      'call': 'np.stack(grating_channels(-0.3, 1.7, 1))',
      'gold_call': 'np.stack(_oracle_grating_channels(-0.3, 1.7, 1))'},
     {'setup': 'def rejects_grazing(fn):\n'
               '    try:\n'
               '        fn(0.0, 0.5, 3)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_grazing(grating_channels)',
      'gold_call': 'rejects_grazing(_oracle_grating_channels)'}]

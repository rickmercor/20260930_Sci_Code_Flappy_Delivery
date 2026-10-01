"""
Compute the paper's relativistic wavelength and interaction constant. Use V=1000*voltage_kv volts, lambda=h/sqrt(2*m0*e*V*(1+e*V/(2*m0*c^2)))*1e10 Å, and sigma=2*pi/(lambda*V)*(m0*c^2+e*V)/(2*m0*c^2+e*V), with h=6.62607015e-34, m0=9.1093837139e-31, e=1.602176634e-19, and c=299792458. Reject nonfinite or nonpositive voltage.

These constants set every transmission and Fresnel phase in the QuScope circuit.

Returns
-------
np.ndarray, (2,) float64 vector [wavelength_A, interaction_constant_rad_per_VA].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relativistic_electron_constants(voltage_kv: float) -> 'np.ndarray':
    """Convert one positive finite accelerating voltage in kV.

Parameters
----------
voltage_kv : float
    Positive finite accelerating voltage, in kV.

Returns
-------
result : np.ndarray
    (2,) float64 vector [wavelength_A, interaction_constant_rad_per_VA].

Notes
-----
Convert one positive finite accelerating voltage in kV.

Returns a float64 ndarray of shape (2,) ordered as
[wavelength_A, interaction_constant_rad_per_VA].

Raises ValueError for nonfinite or nonpositive voltage_kv."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_relativistic_electron_constants(voltage_kv: float) -> 'np.ndarray':
    """Return [wavelength_A, interaction_constant_rad_per_VA]."""
    voltage_kv = float(voltage_kv)
    if not math.isfinite(voltage_kv) or voltage_kv <= 0.0:
        raise ValueError('voltage_kv must be finite and positive')
    h = 6.62607015e-34
    m0 = 9.1093837139e-31
    e = 1.602176634e-19
    c = 299792458.0
    voltage_v = 1000.0 * voltage_kv
    wavelength_a = h / math.sqrt(2.0 * m0 * e * voltage_v * (1.0 + e * voltage_v / (2.0 * m0 * c * c))) * 10000000000.0
    sigma = 2.0 * math.pi / (wavelength_a * voltage_v) * (m0 * c * c + e * voltage_v) / (2.0 * m0 * c * c + e * voltage_v)
    return np.asarray([wavelength_a, sigma], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'relativistic_electron_constants(100.0)',
      'gold_call': '_oracle_relativistic_electron_constants(100.0)'},
     {'setup': '',
      'call': 'relativistic_electron_constants(200.0)',
      'gold_call': '_oracle_relativistic_electron_constants(200.0)'},
     {'setup': '',
      'call': 'relativistic_electron_constants(300.0)',
      'gold_call': '_oracle_relativistic_electron_constants(300.0)'},
     {'setup': '',
      'call': 'relativistic_electron_constants(80.0)',
      'gold_call': '_oracle_relativistic_electron_constants(80.0)'},
     {'setup': '',
      'call': 'relativistic_electron_constants(150.0)',
      'gold_call': '_oracle_relativistic_electron_constants(150.0)'},
     {'setup': 'def _status(fn):\n'
               '    try:\n'
               '        fn(0.0)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_status(relativistic_electron_constants)',
      'gold_call': '_status(_oracle_relativistic_electron_constants)'},
     {'setup': '',
      'call': 'relativistic_electron_constants(400.0)',
      'gold_call': '_oracle_relativistic_electron_constants(400.0)'}]

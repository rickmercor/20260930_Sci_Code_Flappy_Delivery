"""
Solve the source's phase-field balance at a point for the updated phase-field value, given the current crack driving force, the source coefficient produced by the degradation step, the supplied Laplacian of the phase field, and the fracture properties. Use the phase-field distribution function the source selects in Section 2, together with the source coefficient its balance carries; the source argues at length why the other common choice of distribution function would be wrong for this formulation, and that argument fixes which one to use.

Eq. (10) is the stationarity condition of the total energy with respect to the phase-field parameter, and Eq. (22) is its discrete counterpart. The first term comes from the surface energy of Eq. (3) written through the distribution function of Section 2, and the second from the derivative of the degradation function acting on the crack driving force. The balance is linear in the updated value once the driving force is fixed, so it can be solved directly.

Returns
-------
A Python float: the updated phase-field value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phase_field_update(F_hist: float, Gc: float, ell: float, source_coeff: float, lap_phi: float) -> float:
    """Solve the source's phase-field balance at a point for the updated phase-field value,
    given the current crack driving force, the source coefficient produced by the
    degradation step, the supplied Laplacian of the phase field, and the fracture
    properties, using the distribution function the source selects in Section 2.

    Args:
        F_hist: Crack driving force after the irreversibility rule, in GPa, non-negative.
        Gc: Fracture energy release rate.
        ell: Phase-field length scale, in m.
        source_coeff: The positive, phase-field-independent source coefficient returned by
            the degradation step.
        lap_phi: Laplacian of the phase field supplied as data at this increment.

    Returns:
        A Python float: the updated phase-field value.

    Raises:
        ValueError: If F_hist is negative or not finite, if Gc or ell is not positive, if
            source_coeff is not a positive finite number, or if lap_phi is not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_phase_field_update(F_hist: float, Gc: float, ell: float, source_coeff: float, lap_phi: float) -> float:
    if not np.isfinite(F_hist) or F_hist < 0.0:
        raise ValueError("F_hist must be a non-negative finite driving force")
    if Gc <= 0.0 or ell <= 0.0:
        raise ValueError("Gc and ell must be positive")
    if not np.isfinite(source_coeff) or source_coeff <= 0.0:
        raise ValueError("source_coeff must be a positive finite coefficient")
    if not np.isfinite(lap_phi):
        raise ValueError("lap_phi must be finite")
    pre = source_coeff * F_hist
    return float((Gc * ell * lap_phi + pre) / (Gc / ell + pre))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nF_hist = 2.5e-4\nGc = 1.0e-4\nell = 0.05\nsource_coeff = 1.998\nlap_phi = 0.0\n',
         'call': 'phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)',
         'gold_call': '_oracle_phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)'},
        {'setup': 'import numpy as np\nF_hist = 6.0e-4\nGc = 1.0e-4\nell = 0.05\nsource_coeff = 1.998\nlap_phi = 40.0\n',
         'call': 'phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)',
         'gold_call': '_oracle_phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)'},
        {'setup': 'import numpy as np\nF_hist = 0.0\nGc = 2.5e-4\nell = 0.02\nsource_coeff = 1.9\nlap_phi = 12.0\n',
         'call': 'phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)',
         'gold_call': '_oracle_phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi)'},
        {'setup': 'import numpy as np\n# invalid input: a zero fracture energy release rate is not positive and must raise ValueError\nF_hist = 2.5e-4\nGc = 0.0\nell = 0.05\nsource_coeff = 1.998\nlap_phi = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi))',
         'gold_call': '_catches_value_error(lambda: _oracle_phase_field_update(F_hist, Gc, ell, source_coeff, lap_phi))'},
    ]

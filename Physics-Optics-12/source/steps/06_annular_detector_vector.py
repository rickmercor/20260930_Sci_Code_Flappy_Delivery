"""
Transform the exit amplitudes to momentum space with orthonormal FFT2, form normalized intensity |psi(k)|^2/||psi||^2, and sum it over the three consecutive half-open annuli [e0,e1), [e1,e2), [e2,e3). Use theta_mrad=1000*lambda*sqrt(kx^2+ky^2), unshifted fftfreq, ij indexing, and reject an empty mask.

QuScope hardware exposes measurement probabilities, so low-dimensional annular signals are the relevant observable.

Returns
-------
np.ndarray, (3,) real probabilities for the three half-open detector annuli.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def annular_detector_vector(exit_state: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Integrate reciprocal intensity in three consecutive half-open annuli.

Parameters
----------
exit_state : array_like
    Finite nonzero square complex exit amplitudes; no unit-norm precondition.
wavelength_a : float
    Positive finite electron wavelength, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
detector_edges_mrad : array_like
    Four finite increasing nonnegative angles in mrad; each half-open annulus must contain a grid point.

Returns
-------
result : np.ndarray
    (3,) real probabilities for the three half-open detector annuli.

Notes
-----
Integrate reciprocal intensity in three consecutive half-open annuli.

exit_state is a finite, nonzero square complex array; wavelength_a and pixel_size_a are positive; and
detector_edges_mrad contains four increasing edges with nonempty masks.
Probabilities use the full state norm, so the input need not be unit norm.
Invalid sampling, edges or empty masks raise ValueError.
Returns a float ndarray [P0,P1,P2] of shape (3,)."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_annular_detector_vector(exit_state: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Return the three probabilities for consecutive half-open annuli."""
    state = np.asarray(exit_state, dtype=complex)
    edges = np.asarray(detector_edges_mrad, dtype=float)
    wavelength_a = float(wavelength_a)
    pixel_size_a = float(pixel_size_a)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError('exit_state must be square')
    if edges.shape != (4,) or not np.all(np.isfinite(edges)):
        raise ValueError('detector_edges_mrad must have shape (4,)')
    if edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError('detector edges must be strictly increasing and nonnegative')
    if not math.isfinite(wavelength_a) or not math.isfinite(pixel_size_a) or wavelength_a <= 0.0 or (pixel_size_a <= 0.0):
        raise ValueError('invalid sampling')
    frequency = np.fft.fftfreq(state.shape[0], d=pixel_size_a)
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    theta = 1000.0 * wavelength_a * np.sqrt(kx * kx + ky * ky)
    intensity = np.abs(np.fft.fft2(state, norm='ortho')) ** 2 / float(np.vdot(state, state).real)
    values = []
    for (lo, hi) in zip(edges[:-1], edges[1:]):
        mask = (theta >= lo) & (theta < hi)
        if not np.any(mask):
            raise ValueError('every detector annulus must contain a grid point')
        values.append(float(np.sum(intensity[mask])))
    return np.asarray(values, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
s=np.ones((8,8),complex)/8
import copy
s_gold = copy.deepcopy(s)""",
            "call": "annular_detector_vector(s,0.03701436611487085,0.8,np.array([0.0,5.8,8.2,13.0]))",
            "gold_call": "_oracle_annular_detector_vector(s_gold, 0.03701436611487085, 0.8, np.array([0.0, 5.8, 8.2, 13.0]))"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(16,0.4,1,3)
p=amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,10.0,0.0)
w=quantum_multislice_exit_state(s,p,0.025079340436272274,0.0007288401043940304,1.8,3,0.4)
s_gold=_oracle_build_grouped_strong_scattering_stack(16,0.4,1,3)
p_gold=_oracle_amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,10.0,0.0)
w_gold=_oracle_quantum_multislice_exit_state(s_gold,p_gold,0.025079340436272274,0.0007288401043940304,1.8,3,0.4)""",
            "call": "annular_detector_vector(w,0.025079340436272274,0.4,[1.0,6.0,14.0,24.0])",
            "gold_call": "_oracle_annular_detector_vector(w_gold, 0.025079340436272274, 0.4, [1.0, 6.0, 14.0, 24.0])"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(32,0.2,0,1)
p=amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,18.0,30.0)
w=quantum_multislice_exit_state(s,p,0.01968748899648993,0.0006526161423885242,1.8,1,0.2)
s_gold=_oracle_build_grouped_strong_scattering_stack(32,0.2,0,1)
p_gold=_oracle_amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,18.0,30.0)
w_gold=_oracle_quantum_multislice_exit_state(s_gold,p_gold,0.01968748899648993,0.0006526161423885242,1.8,1,0.2)""",
            "call": "float(annular_detector_vector(w,0.01968748899648993,0.2,[4.0,10.0,16.0,22.0])[2])",
            "gold_call": "float(_oracle_annular_detector_vector(w_gold, 0.01968748899648993, 0.2, [4.0, 10.0, 16.0, 22.0])[2])"
        },
        {
            "setup": """import numpy as np
s=np.zeros((16,16),complex)
s[4,9]=1
import copy
s_gold = copy.deepcopy(s)""",
            "call": "annular_detector_vector(s,0.025079340436272274,0.4,[0.0,4.0,8.0,16.0])",
            "gold_call": "_oracle_annular_detector_vector(s_gold, 0.025079340436272274, 0.4, [0.0, 4.0, 8.0, 16.0])"
        },
        {
            "setup": """import numpy as np
s=np.zeros((8,8),complex)
s[0,0]=1
import copy
s_gold = copy.deepcopy(s)""",
            "call": "annular_detector_vector(s,0.03,0.8,[0.0,6.0,12.0,18.0])",
            "gold_call": "_oracle_annular_detector_vector(s_gold, 0.03, 0.8, [0.0, 6.0, 12.0, 18.0])"
        },
        {
            "setup": """import numpy as np
x=np.arange(64).reshape(8,8)
s=np.exp(2j*np.pi*x/64)/8
import copy
s_gold = copy.deepcopy(s)""",
            "call": "annular_detector_vector(s,0.037,0.8,[0.0,5.8,8.2,13.0])",
            "gold_call": "_oracle_annular_detector_vector(s_gold, 0.037, 0.8, [0.0, 5.8, 8.2, 13.0])"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(32,0.2,1,3)
p=amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,10.0,-15.0)
w=quantum_multislice_exit_state(s,p,0.025079340436272274,0.0007288401043940304,1.8,3,0.2)
s_gold=_oracle_build_grouped_strong_scattering_stack(32,0.2,1,3)
p_gold=_oracle_amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,10.0,-15.0)
w_gold=_oracle_quantum_multislice_exit_state(s_gold,p_gold,0.025079340436272274,0.0007288401043940304,1.8,3,0.2)""",
            "call": "float(annular_detector_vector(w,0.025079340436272274,0.2,[1.0,6.0,14.0,24.0])[0])",
            "gold_call": "float(_oracle_annular_detector_vector(w_gold, 0.025079340436272274, 0.2, [1.0, 6.0, 14.0, 24.0])[0])"
        },
        {
            "setup": """import numpy as np
s=3*np.ones((8,8),complex)/8
import copy
s_gold = copy.deepcopy(s)""",
            "call": "annular_detector_vector(s,0.03701436611487085,0.8,np.array([0.0,5.8,8.2,13.0]))",
            "gold_call": "_oracle_annular_detector_vector(s_gold, 0.03701436611487085, 0.8, np.array([0.0, 5.8, 8.2, 13.0]))"
        }
    ]

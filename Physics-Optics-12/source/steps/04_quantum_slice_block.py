"""
Apply the exact position-space phase grating D[exp(i*sigma*V)]. If propagate_after is True, also apply the separable orthonormal forward transform F@state@F.T with F[a,b]=exp(-2*pi*i*a*b/N)/sqrt(N), the reciprocal phase exp(-i*pi*lambda*Delta_z*(kx^2+ky^2)) on the unshifted ij-indexed grid, and the inverse transform. If False, omit the whole propagation segment. The public optional flag makes interior and terminal STEM gratings explicit; projected potentials are in V Å.

The source STEM builder has one phase grating per retained slice but Fresnel propagation only between consecutive gratings.

Returns
-------
np.ndarray, (N,N) complex amplitudes after transmission and the optional Fresnel segment.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantum_slice_block(state: 'np.ndarray', projected_potential: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float, propagate_after: bool=True) -> 'np.ndarray':
    """Apply an exact phase grating and an optional following Fresnel segment.

Parameters
----------
state : array_like
    Finite complex (N,N) state on a power-of-two grid.
projected_potential : array_like
    Finite real (N,N) projected potential, in V Å.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Positive finite interaction constant in rad/(V Å).
propagation_distance_a : float
    Positive finite inter-grating propagation distance, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
propagate_after : bool
    Whether to include the free-propagation segment after this transmission.

Returns
-------
result : np.ndarray
    (N,N) complex amplitudes after transmission and the optional Fresnel segment.

Notes
-----
Apply an exact phase grating and an optional following Fresnel segment.

state and projected_potential are equal (N,N) power-of-two arrays; potential
is projected in V Å. wavelength_a, propagation_distance_a, pixel_size_a
are positive finite Å scalars, and interaction_constant is positive finite
rad/(V Å). Boolean propagate_after=True includes the following segment;
False returns immediately after the grating, as for the final STEM slice.
Returns a complex ndarray of the same (N,N) shape as state. Invalid inputs raise ValueError."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_quantum_slice_block(state: 'np.ndarray', projected_potential: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float, propagate_after: bool=True) -> 'np.ndarray':
    """Apply a phase grating and, when requested, an inter-grating Fresnel segment."""
    wave = np.asarray(state, dtype=complex).copy()
    potential = np.asarray(projected_potential, dtype=float)
    scalars = [float(wavelength_a), float(interaction_constant), float(propagation_distance_a), float(pixel_size_a)]
    if wave.ndim != 2 or wave.shape[0] != wave.shape[1] or wave.shape != potential.shape:
        raise ValueError('state and potential must be equal square arrays')
    if wave.shape[0] < 2 or wave.shape[0] & wave.shape[0] - 1:
        raise ValueError('linear grid size must be a power of two')
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(potential)):
        raise ValueError('state and potential must be finite')
    if any((not math.isfinite(x) or x <= 0.0 for x in scalars)):
        raise ValueError('physical scalars must be finite and positive')
    if not isinstance(propagate_after, (bool, np.bool_)):
        raise ValueError('propagate_after must be boolean')
    wave *= np.exp(1j * scalars[1] * potential)
    if not propagate_after:
        return wave
    n = wave.shape[0]
    indices = np.arange(n, dtype=float)
    qft = np.exp(-2j * np.pi * np.outer(indices, indices) / n) / np.sqrt(n)
    frequency = np.fft.fftfreq(n, d=scalars[3])
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    propagator = np.exp(-1j * np.pi * scalars[0] * scalars[2] * (kx * kx + ky * ky))
    reciprocal = qft @ wave @ qft.T
    return np.asarray(qft.conj().T @ (reciprocal * propagator) @ qft.conj(), dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
s=np.zeros((4,4),complex); s[0,0]=1
v=np.zeros((4,4))
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(abs(quantum_slice_block(s,v,0.025,0.0007,1.8,0.4)[0,0]))",
            "gold_call": "float(abs(_oracle_quantum_slice_block(s_gold, v_gold, 0.025, 0.0007, 1.8, 0.4)[0, 0]))"
        },
        {
            "setup": """import numpy as np
s=np.ones((8,8),complex)/8
v=np.arange(64,dtype=float).reshape(8,8)
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(quantum_slice_block(s,v,0.025079340436272274,0.0007288401043940304,3.6,0.8)[2,5].real)",
            "gold_call": "float(_oracle_quantum_slice_block(s_gold, v_gold, 0.025079340436272274, 0.0007288401043940304, 3.6, 0.8)[2, 5].real)"
        },
        {
            "setup": """import numpy as np
s=np.zeros((16,16),complex)
s[3,7]=1
s_gold=s.copy()
v=build_grouped_strong_scattering_stack(16,0.4,1,6)[0]
v_gold=_oracle_build_grouped_strong_scattering_stack(16,0.4,1,6)[0]""",
            "call": "float(abs(quantum_slice_block(s,v,0.01968748899648993,0.0006526161423885242,10.8,0.4)[11,4]))",
            "gold_call": "float(abs(_oracle_quantum_slice_block(s_gold, v_gold, 0.01968748899648993, 0.0006526161423885242, 10.8, 0.4)[11, 4]))"
        },
        {
            "setup": """import numpy as np
s=np.eye(8,dtype=complex)/np.sqrt(8)
v=np.flipud(np.arange(64,dtype=float).reshape(8,8))
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(quantum_slice_block(s,v,0.03701436611487085,0.0009243958175400269,5.4,0.8)[6,1].imag)",
            "gold_call": "float(_oracle_quantum_slice_block(s_gold, v_gold, 0.03701436611487085, 0.0009243958175400269, 5.4, 0.8)[6, 1].imag)"
        },
        {
            "setup": """import numpy as np
s=np.ones((4,4),complex)/4
v=np.full((4,4),700.0)
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(quantum_slice_block(s,v,0.03,0.0008,1.8,1.6)[3,2].real)",
            "gold_call": "float(_oracle_quantum_slice_block(s_gold, v_gold, 0.03, 0.0008, 1.8, 1.6)[3, 2].real)"
        },
        {
            "setup": """import numpy as np
s=np.eye(16,dtype=complex)/4
v=np.zeros((16,16))
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(abs(quantum_slice_block(s,v,0.0197,0.00065,10.8,0.4)[9,9]))",
            "gold_call": "float(abs(_oracle_quantum_slice_block(s_gold, v_gold, 0.0197, 0.00065, 10.8, 0.4)[9, 9]))"
        },
        {
            "setup": """import numpy as np
s=np.zeros((8,8),complex)
s[7,0]=1
v=np.arange(64,dtype=float).reshape(8,8)**2
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "float(quantum_slice_block(s,v,0.025,0.00073,3.6,0.8)[0,7].imag)",
            "gold_call": "float(_oracle_quantum_slice_block(s_gold, v_gold, 0.025, 0.00073, 3.6, 0.8)[0, 7].imag)"
        },
        {
            "setup": """import numpy as np
s=np.eye(4,dtype=complex)/2
v=np.arange(16,dtype=float).reshape(4,4)
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "quantum_slice_block(s,v,0.025,0.0007,1.8,0.4,propagate_after=False)",
            "gold_call": "_oracle_quantum_slice_block(s_gold, v_gold, 0.025, 0.0007, 1.8, 0.4, propagate_after=False)"
        },
        {
            "setup": """import numpy as np
s=np.zeros((8,8),complex)
s[2,5]=1
v=np.zeros((8,8))
import copy
s_gold = copy.deepcopy(s)
v_gold = copy.deepcopy(v)""",
            "call": "quantum_slice_block(s,v,0.03,0.0008,8.0,0.8,propagate_after=False)",
            "gold_call": "_oracle_quantum_slice_block(s_gold, v_gold, 0.03, 0.0008, 8.0, 0.8, propagate_after=False)"
        }
    ]

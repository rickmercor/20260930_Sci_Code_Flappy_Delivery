"""
Execute quantum_slice_block once for each consecutive grouped potential, in order. Use propagation distance base_slice_thickness_a*group_factor and propagate_after=True only when another retained grating follows; use False for the final grating. The stack contains exactly 6/group_factor slices. Return the immediate post-grating STEM exit, without an objective lens or final free-propagation segment.

This follows the released STEM probe builder rather than the lens-inclusive CTEM circuit.

Returns
-------
np.ndarray, (N,N) complex amplitudes immediately after the final retained grating.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantum_multislice_exit_state(grouped_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, base_slice_thickness_a: float, group_factor: int, pixel_size_a: float) -> 'np.ndarray':
    """Propagate an incident state through every grouped slice in order.

Parameters
----------
grouped_stack : array_like
    Finite real (s,N,N) projected potentials in V Å, with s*group_factor=6.
incident_state : array_like
    Finite complex (N,N) input amplitudes matching the potential grid.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Positive finite interaction constant in rad/(V Å).
base_slice_thickness_a : float
    Positive finite thickness of one base slice, in Å.
group_factor : int
    Number of consecutive base slices per retained grating; one of 1, 2, 3, 6.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.

Returns
-------
result : np.ndarray
    (N,N) complex amplitudes immediately after the final retained grating.

Notes
-----
Propagate an incident state through every grouped slice in order.

grouped_stack is (s,n,n), incident_state is (n,n), and s*group_factor
equals six. Propagation is applied only between gratings, not after the
final one; scalar arguments are positive and finite. Returns the final
complex ndarray with the same (n,n) shape as incident_state."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_quantum_multislice_exit_state(grouped_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, base_slice_thickness_a: float, group_factor: int, pixel_size_a: float) -> 'np.ndarray':
    """Execute one QuScope quantum slice block for every grouped slice."""
    stack = np.asarray(grouped_stack, dtype=float)
    state = np.asarray(incident_state, dtype=complex).copy()
    group_factor = int(group_factor)
    if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1] != stack.shape[2]:
        raise ValueError('grouped_stack must have shape (s,n,n)')
    if state.shape != stack.shape[1:]:
        raise ValueError('incident state shape mismatch')
    if group_factor not in (1, 2, 3, 6) or stack.shape[0] * group_factor != 6:
        raise ValueError('grouping must represent exactly six base slices')
    distance = float(base_slice_thickness_a) * group_factor
    for (j, potential) in enumerate(stack):
        state = _oracle_quantum_slice_block(state, potential, wavelength_a, interaction_constant, distance, pixel_size_a, propagate_after=j < stack.shape[0] - 1)
    return np.asarray(state, dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """s=build_grouped_strong_scattering_stack(8,0.8,0,6)
p=amplitude_encoded_stem_probe(8,0.8,0.03701436611487085,10.0,0.0)
s_gold=_oracle_build_grouped_strong_scattering_stack(8,0.8,0,6)
p_gold=_oracle_amplitude_encoded_stem_probe(8,0.8,0.03701436611487085,10.0,0.0)""",
            "call": "float(abs(quantum_multislice_exit_state(s,p,0.03701436611487085,0.0009243958175400269,1.8,6,0.8)[2,3]))",
            "gold_call": "float(abs(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.03701436611487085, 0.0009243958175400269, 1.8, 6, 0.8)[2, 3]))"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(16,0.4,1,3)
p=amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,18.0,-30.0)
s_gold=_oracle_build_grouped_strong_scattering_stack(16,0.4,1,3)
p_gold=_oracle_amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,18.0,-30.0)""",
            "call": "float(quantum_multislice_exit_state(s,p,0.025079340436272274,0.0007288401043940304,1.8,3,0.4)[7,9].imag)",
            "gold_call": "float(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.025079340436272274, 0.0007288401043940304, 1.8, 3, 0.4)[7, 9].imag)"
        },
        {
            "setup": """import numpy as np
s=build_grouped_strong_scattering_stack(32,0.2,1,1)
p=amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,22.0,30.0)
s_gold=_oracle_build_grouped_strong_scattering_stack(32,0.2,1,1)
p_gold=_oracle_amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,22.0,30.0)""",
            "call": "float(np.linalg.norm(quantum_multislice_exit_state(s,p,0.01968748899648993,0.0006526161423885242,1.8,1,0.2)))",
            "gold_call": "float(np.linalg.norm(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.01968748899648993, 0.0006526161423885242, 1.8, 1, 0.2)))"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(16,0.4,0,2)
p=amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,14.0,15.0,0.05,(0.18,0.0))
s_gold=_oracle_build_grouped_strong_scattering_stack(16,0.4,0,2)
p_gold=_oracle_amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,14.0,15.0,0.05,(0.18,0.0))""",
            "call": "float(abs(quantum_multislice_exit_state(s,p,0.025079340436272274,0.0007288401043940304,1.8,2,0.4)[5,12]))",
            "gold_call": "float(abs(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.025079340436272274, 0.0007288401043940304, 1.8, 2, 0.4)[5, 12]))"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(8,0.8,1,3)
p=amplitude_encoded_stem_probe(8,0.8,0.03,12.0,-15.0)
s_gold=_oracle_build_grouped_strong_scattering_stack(8,0.8,1,3)
p_gold=_oracle_amplitude_encoded_stem_probe(8,0.8,0.03,12.0,-15.0)""",
            "call": "float(abs(quantum_multislice_exit_state(s,p,0.03,0.0008,1.8,3,0.8)[6,6]))",
            "gold_call": "float(abs(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.03, 0.0008, 1.8, 3, 0.8)[6, 6]))"
        },
        {
            "setup": """import numpy as np
s=build_grouped_strong_scattering_stack(16,0.4,0,6)
p=amplitude_encoded_stem_probe(16,0.4,0.025,10.0,45.0)
s_gold=_oracle_build_grouped_strong_scattering_stack(16,0.4,0,6)
p_gold=_oracle_amplitude_encoded_stem_probe(16,0.4,0.025,10.0,45.0)""",
            "call": "float(np.linalg.norm(quantum_multislice_exit_state(s,p,0.025,0.00073,1.8,6,0.4))**2)",
            "gold_call": "float(np.linalg.norm(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.025, 0.00073, 1.8, 6, 0.4)) ** 2)"
        },
        {
            "setup": """s=build_grouped_strong_scattering_stack(32,0.2,1,2)
p=amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,13.0,-45.0,0.05,(0.31,-0.13))
s_gold=_oracle_build_grouped_strong_scattering_stack(32,0.2,1,2)
p_gold=_oracle_amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,13.0,-45.0,0.05,(0.31,-0.13))""",
            "call": "float(quantum_multislice_exit_state(s,p,0.01968748899648993,0.0006526161423885242,1.8,2,0.2)[13,19].real)",
            "gold_call": "float(_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.01968748899648993, 0.0006526161423885242, 1.8, 2, 0.2)[13, 19].real)"
        },
        {
            "setup": """import numpy as np
s=np.arange(64,dtype=float).reshape(1,8,8)
p=np.eye(8,dtype=complex)/np.sqrt(8)
import copy
p_gold = copy.deepcopy(p)
s_gold = copy.deepcopy(s)""",
            "call": "quantum_multislice_exit_state(s,p,0.03,0.0008,1.8,6,0.8)",
            "gold_call": "_oracle_quantum_multislice_exit_state(s_gold, p_gold, 0.03, 0.0008, 1.8, 6, 0.8)"
        }
    ]

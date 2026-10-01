"""
Cost the released STEM probe-circuit body for an N-by-N grid and s retained gratings. With q=log2(N), nq=2q, count s transmission diagonals and s-1 propagation diagonals, each at 2^nq-2 CX, and 2(s-1) separable two-dimensional Fourier transforms. A full q-qubit register QFT costs q(q-1)+3*floor(q/2) CX, including swaps, and each two-dimensional transform acts on both registers. This all-to-all analytic synthesis convention reproduces the paper's STEM resource examples; state preparation, detector-plane readout and device routing are outside the boundary. Return [nq,2s-1,2(s-1),total_CX].

QuScope Table III distinguishes STEM probe bodies from CTEM circuits. The one-grating limit has one diagonal and no inter-grating propagation.

Returns
-------
np.ndarray, (4,) real vector [logical_qubits,diagonal_gate_count,separable_2D_QFT_count,total_CX].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transpiled_multislice_resources(n: int, slice_count: int) -> 'np.ndarray':
    """Count the all-to-all STEM probe-circuit body, with no final propagation.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
slice_count : int
    Positive integer retained-grating count.

Returns
-------
result : np.ndarray
    (4,) real vector [logical_qubits,diagonal_gate_count,separable_2D_QFT_count,total_CX].

Notes
-----
Count the all-to-all STEM probe-circuit body, with no final propagation.

n is the linear grid dimension and slice_count is the positive retained
grating count. State preparation and detector-plane readout are outside
this cost boundary. Returns a float ndarray of shape (4,) ordered as
[logical_qubits,diagonal_gate_count,separable_2D_QFT_count,total_CX]."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_transpiled_multislice_resources(n: int, slice_count: int) -> 'np.ndarray':
    """Return [logical_qubits, diagonal_gates, 2-D QFTs, total_CX]."""
    n = int(n)
    slice_count = int(slice_count)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if slice_count < 1:
        raise ValueError('slice_count must be positive')
    register_qubits = int(round(math.log2(n)))
    logical_qubits = 2 * register_qubits
    diagonal_gates = 2 * slice_count - 1
    qft2_transforms = 2 * (slice_count - 1)
    arbitrary_diagonal_cx = 2 ** logical_qubits - 2
    one_register_qft_cx = register_qubits * (register_qubits - 1) + 3 * (register_qubits // 2)
    total_cx = diagonal_gates * arbitrary_diagonal_cx + qft2_transforms * 2 * one_register_qft_cx
    return np.asarray([logical_qubits, diagonal_gates, qft2_transforms, total_cx], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'transpiled_multislice_resources(8,3)',
      'gold_call': '_oracle_transpiled_multislice_resources(8,3)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(16,3)',
      'gold_call': '_oracle_transpiled_multislice_resources(16,3)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(32,2)',
      'gold_call': '_oracle_transpiled_multislice_resources(32,2)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(64,1)',
      'gold_call': '_oracle_transpiled_multislice_resources(64,1)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(8,6)',
      'gold_call': '_oracle_transpiled_multislice_resources(8,6)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(32,6)',
      'gold_call': '_oracle_transpiled_multislice_resources(32,6)'},
     {'setup': '',
      'call': 'transpiled_multislice_resources(128,3)',
      'gold_call': '_oracle_transpiled_multislice_resources(128,3)'},
     {'setup': 'import numpy as np',
      'call': 'transpiled_multislice_resources(8,1)',
      'gold_call': '_oracle_transpiled_multislice_resources(8,1)'},
     {'setup': 'import numpy as np',
      'call': 'transpiled_multislice_resources(16,1)',
      'gold_call': '_oracle_transpiled_multislice_resources(16,1)'},
     {'setup': 'import numpy as np',
      'call': 'transpiled_multislice_resources(16,2)',
      'gold_call': '_oracle_transpiled_multislice_resources(16,2)'}]

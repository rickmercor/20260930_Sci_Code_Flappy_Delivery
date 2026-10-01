"""
Construct the ordered rank-two operator-Schmidt factors for every two-site Pauli-coupling gate in the benchmark circuit.

For an axis \(P\in\{X,Y,Z\}\), the gate \(\exp(i\theta P\otimes P)\) is represented as two factor pairs: \(V^0=W^0=\sqrt{\cos\theta},I\), \(V^1=i\sqrt{\sin\theta},P\), and \(W^1=\sqrt{\sin\theta},P\).  The factor axes are ordered as ``(event, Schmidt term, output, input)`` and the event order is chronological.

Returns
-------
Return `(left_gate_factors, right_gate_factors)`, two complex128 arrays of shape `(L,2,2,2)` in chronological `(event,term,output,input)` order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_operator_schmidt_factors(
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return left and right rank-two gate factors of shape ``(L,2,2,2)``.

    ``pauli_axes`` uses ``0`` for X, ``1`` for Y, and ``2`` for Z.  Angles are
    real radians strictly between zero and pi/2.  Raise ``ValueError`` for
    mismatched shapes, invalid axis codes, nonfinite data, or invalid angles.
    """
    return left_gate_factors, right_gate_factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_operator_schmidt_factors(
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    axes = np.asarray(pauli_axes)
    theta = np.asarray(angles, dtype=np.float64)
    if axes.ndim != 1 or theta.ndim != 1 or axes.shape != theta.shape or axes.size == 0:
        raise ValueError("axes and angles must be matching nonempty vectors")
    if axes.dtype.kind not in "iu" or np.any((axes < 0) | (axes > 2)):
        raise ValueError("Pauli axes must be integer codes 0, 1, or 2")
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0) or np.any(theta >= np.pi / 2.0):
        raise ValueError("angles must be finite and lie strictly between zero and pi/2")
    paulis = np.array([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    left = np.empty((axes.size, 2, 2, 2), dtype=np.complex128)
    right = np.empty_like(left)
    for event, (axis, angle) in enumerate(zip(axes.astype(np.int64), theta)):
        left[event, 0] = np.sqrt(np.cos(angle)) * identity
        right[event, 0] = np.sqrt(np.cos(angle)) * identity
        left[event, 1] = 1.0j * np.sqrt(np.sin(angle)) * paulis[axis]
        right[event, 1] = np.sqrt(np.sin(angle)) * paulis[axis]
        gate = np.einsum("bos,bqt->oqst", left[event], right[event]).reshape(4, 4)
        target = np.cos(angle) * np.eye(4) + 1.0j * np.sin(angle) * np.kron(paulis[axis], paulis[axis])
        if np.linalg.norm(gate - target) > 1.0e-12:
            raise ValueError("operator-Schmidt factors do not reconstruct the gate")
    return left, right

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = "import numpy as np\ndef _pack(value):\n    parts=[]\n    def _visit(item):\n        if isinstance(item,(tuple,list)):\n            parts.append(np.array([len(item)],dtype=np.complex128))\n            for child in item: _visit(child)\n        else:\n            arr=np.asarray(item); parts.append(np.asarray([arr.ndim,*arr.shape],dtype=np.complex128)); parts.append(arr.astype(np.complex128).ravel())\n    _visit(value)\n    return np.concatenate(parts)\n"
    return [
        {"setup": pack + "pauli_axes=np.array([0,1,2,1,2,0],dtype=np.int64)\nangles=np.array([.31,.27,.23,.41,.35,.29])", "call": "_pack(build_operator_schmidt_factors(pauli_axes, angles))", "gold_call": "_pack(_oracle_build_operator_schmidt_factors(pauli_axes, angles))", "tol": 1e-12},
        {"setup": pack + "pauli_axes=np.array([2],dtype=np.int64)\nangles=np.array([1e-7])", "call": "_pack(build_operator_schmidt_factors(pauli_axes, angles))", "gold_call": "_pack(_oracle_build_operator_schmidt_factors(pauli_axes, angles))", "tol": 1e-12},
        {"setup": pack + "pauli_axes=np.array([1,0,2],dtype=np.int64)\nangles=np.array([1.2,.66,.11])", "call": "_pack(build_operator_schmidt_factors(pauli_axes, angles))", "gold_call": "_pack(_oracle_build_operator_schmidt_factors(pauli_axes, angles))", "tol": 1e-12},
        {"setup": "import numpy as np\npauli_axes=np.array([3],dtype=np.int64)\nangles=np.array([.2])\ndef candidate_result():\n    try:\n        build_operator_schmidt_factors(pauli_axes,angles)\n        return 0\n    except ValueError:\n        return 1\ndef reference_result():\n    try:\n        _oracle_build_operator_schmidt_factors(pauli_axes,angles)\n        return 0\n    except ValueError:\n        return 1", "call": "candidate_result()", "gold_call": "reference_result()"},
    ]

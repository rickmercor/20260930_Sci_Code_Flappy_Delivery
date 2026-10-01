"""
Form the complete local post-gate tensor and its canonicalized matrix, then compute the deterministic full singular-value decomposition used by the projector construction.

For neighboring right-canonical tensors, the gate-factor contraction gives \(C_{a\sigma',\tau'd}\).  The canonicalized tensor is \(\Theta_{a\sigma',\tau'd}=\lambda_{i-1,a}C_{a\sigma',\tau'd}\), with row order ``(left virtual, left output)`` and column order ``(right output, right virtual)``.  All four singular modes are retained, ordered nonincreasingly, and fixed by the prescribed maximum-magnitude pivot phase convention.

Returns
-------
Return `(C, Theta, X, s, Yh)`: four complex128 `(4,4)` matrices around one nonincreasing float64 spectrum `s` of shape `(4,)`, under the frozen phase convention.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_local_tebd_svd(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return ``(C, Theta, X, s, Yh)`` for one internal rank-two bond.

    ``C``, ``Theta``, ``X``, and ``Yh`` are complex ``(4,4)`` arrays and ``s``
    is a real ``(4,)`` array.  All modes must exceed ``1e-6*s[0]`` and adjacent
    gaps must exceed ``1e-10*s[0]``.  Raise ``ValueError`` when the local input
    shapes or these complete-spectrum domain conditions are violated.
    """
    return post_gate_tensor, canonicalized_tensor, left_singular_vectors, singular_values, right_singular_vectors_h

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg


def _oracle_compute_local_tebd_svd(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    left = np.asarray(left_tensor, dtype=np.complex128)
    right = np.asarray(right_tensor, dtype=np.complex128)
    lam = np.asarray(left_bond_singular_values, dtype=np.float64)
    gate_left = np.asarray(left_gate_factors, dtype=np.complex128)
    gate_right = np.asarray(right_gate_factors, dtype=np.complex128)
    if left.shape != (2, 2, 2) or right.shape != (2, 2, 2):
        raise ValueError("local tensors must both have shape (2,2,2)")
    if lam.shape != (2,) or gate_left.shape != (2, 2, 2) or gate_right.shape != (2, 2, 2):
        raise ValueError("invalid singular-value or gate-factor shape")
    if any(not np.all(np.isfinite(x)) for x in (left, right, lam, gate_left, gate_right)):
        raise ValueError("all inputs must be finite")
    if np.any(lam <= 0.0) or lam[0] < lam[1]:
        raise ValueError("left-bond singular values must be positive and nonincreasing")
    gate = np.einsum("bos,bqt->oqst", gate_left, gate_right)
    post_gate = np.einsum("oqst,sac,tcd->aoqd", gate, left, right).reshape(4, 4)
    theta = (lam[:, None, None, None] * post_gate.reshape(2, 2, 2, 2)).reshape(4, 4)
    x, singular_values, yh = scipy.linalg.svd(
        theta, full_matrices=False, lapack_driver="gesvd"
    )
    x, yh = _phase_fix_left_columns(x, yh)
    if singular_values[0] <= 0.0 or np.any(singular_values <= 1.0e-6 * singular_values[0]):
        raise ValueError("the complete four-mode local spectrum is required")
    if np.any(singular_values[:-1] - singular_values[1:] <= 1.0e-10 * singular_values[0]):
        raise ValueError("active local singular values must be nondegenerate")
    return (
        np.asarray(post_gate, dtype=np.complex128),
        np.asarray(theta, dtype=np.complex128),
        np.asarray(x, dtype=np.complex128),
        np.asarray(singular_values, dtype=np.float64),
        np.asarray(yh, dtype=np.complex128),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    # Evaluate each side lazily with its own implementation chain and fresh inputs.
    setup_1 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd):
    import numpy as np

    def _pack(value):
        parts = []
        for item in value:
            arr = np.asarray(item)
            parts.append(np.asarray([arr.ndim, *arr.shape], dtype=np.complex128))
            parts.append(arr.astype(np.complex128).ravel())
        return np.concatenate(parts)
    dims = (1, 2, 2, 2, 2, 2, 1)
    raw = []
    for n in range(6):
        t = np.empty((2, dims[n], dims[n + 1]), dtype=np.complex128)
        for sig in range(2):
            for a in range(dims[n]):
                for b in range(dims[n + 1]):
                    x = 11 * (n + 1) + 7 * (sig + 1) + 5 * (a + 1) + 3 * (b + 1) + (n + 1) * (sig + 1) * (a + b + 2)
                    t[sig, a, b] = np.sin(0.37 * x) + 0.31 * np.cos(0.19 * (x * x + 3 * a + 5 * b)) + 1j * (np.cos(0.29 * (x + 2 * a * b)) + 0.23 * np.sin(0.41 * (x * x + sig + b)))
        raw.append(t)
    canonical, spectra = canonicalize_initial_mps(tuple(raw))
    gl, gr = build_operator_schmidt_factors(np.array([0]), np.array([0.31]))
    left_tensor = canonical[1]
    right_tensor = canonical[2]
    left_bond_singular_values = spectra[0]
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]
    return _pack(compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors))
'''
    setup_2 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd):
    import numpy as np

    def _pack(value):
        parts = []
        for item in value:
            arr = np.asarray(item)
            parts.append(np.asarray([arr.ndim, *arr.shape], dtype=np.complex128))
            parts.append(arr.astype(np.complex128).ravel())
        return np.concatenate(parts)
    dims = (1, 2, 2, 2, 2, 2, 1)
    raw = []
    for n in range(6):
        t = np.empty((2, dims[n], dims[n + 1]), dtype=np.complex128)
        for sig in range(2):
            for a in range(dims[n]):
                for b in range(dims[n + 1]):
                    x = 11 * (n + 1) + 7 * (sig + 1) + 5 * (a + 1) + 3 * (b + 1) + (n + 1) * (sig + 1) * (a + b + 2)
                    t[sig, a, b] = np.sin(0.37 * x) + 0.31 * np.cos(0.19 * (x * x + 3 * a + 5 * b)) + 1j * (np.cos(0.29 * (x + 2 * a * b)) + 0.23 * np.sin(0.41 * (x * x + sig + b)))
        raw.append(t)
    canonical, spectra = canonicalize_initial_mps(tuple(raw))
    gl, gr = build_operator_schmidt_factors(np.array([2]), np.array([0.67]))
    left_tensor = canonical[2]
    right_tensor = canonical[3]
    left_bond_singular_values = spectra[1]
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]
    return _pack(compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors))
'''
    setup_3 = '''def _evaluate(build_operator_schmidt_factors, compute_local_tebd_svd):
    import numpy as np
    left_tensor = np.zeros((2, 2, 2), dtype=np.complex128)
    right_tensor = np.zeros((2, 2, 2), dtype=np.complex128)
    left_tensor[0] = np.eye(2)
    right_tensor[0] = np.eye(2)
    left_bond_singular_values = np.array([0.8, 0.6])
    gl, gr = build_operator_schmidt_factors(np.array([0]), np.array([0.31]))
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]

    def candidate_result():
        try:
            compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors)
            return 0
        except ValueError:
            return 1
    return candidate_result()
'''
    return [
        {
            "setup": setup_1,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_compute_local_tebd_svd)',
            "tol": 1e-10,
        },
        {
            "setup": setup_2,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_compute_local_tebd_svd)',
            "tol": 1e-10,
        },
        {
            "setup": setup_3,
            "call": '_evaluate(build_operator_schmidt_factors, compute_local_tebd_svd)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_compute_local_tebd_svd)',
        },
    ]

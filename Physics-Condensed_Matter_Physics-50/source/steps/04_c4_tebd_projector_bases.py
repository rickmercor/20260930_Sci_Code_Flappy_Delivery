"""
Construct the left and right TEBD projector bases that reproduce the complete local singular decomposition before subspace selection.

The source construction combines the operator-Schmidt gate factors, the neighboring right-canonical tensors, and the complete local SVD.  The left basis \(Q_L^b\) contracts the right tensor with the right singular vectors, while \(Q_R^b\) contracts the left tensor with the left singular vectors, the preceding bond spectrum, and the inverse local singular values.  Their axes are fixed as `$Q_L[b, old_bond, mode]$` and `$Q_R[b, mode, old_bond]$`.

Returns
-------
Return `(Q_L, Q_R)` as complex128 arrays of shapes `(2,2,4)` and `(2,4,2)` with axes `(term,old_bond,mode)` and `(term,mode,old_bond)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_tebd_projector_bases(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_singular_vectors: "np.ndarray",
    singular_values: "np.ndarray",
    right_singular_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(Q_L, Q_R)`` with shapes ``(2,2,4)`` and ``(2,4,2)``.

    The inputs must describe a valid nondegenerate complete four-mode local SVD
    on an internal rank-two bond.  No mode is truncated in this construction.
    """
    return left_projector_basis, right_projector_basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_tebd_projector_bases(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_singular_vectors: "np.ndarray",
    singular_values: "np.ndarray",
    right_singular_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    left = np.asarray(left_tensor, dtype=np.complex128)
    right = np.asarray(right_tensor, dtype=np.complex128)
    lam = np.asarray(left_bond_singular_values, dtype=np.float64)
    gate_left = np.asarray(left_gate_factors, dtype=np.complex128)
    gate_right = np.asarray(right_gate_factors, dtype=np.complex128)
    x = np.asarray(left_singular_vectors, dtype=np.complex128)
    s = np.asarray(singular_values, dtype=np.float64)
    yh = np.asarray(right_singular_vectors_h, dtype=np.complex128)
    expected = ((2, 2, 2), (2, 2, 2), (2,), (2, 2, 2), (2, 2, 2), (4, 4), (4,), (4, 4))
    actual = (left.shape, right.shape, lam.shape, gate_left.shape, gate_right.shape, x.shape, s.shape, yh.shape)
    if actual != expected:
        raise ValueError("projector-basis inputs have incompatible shapes")
    if any(not np.all(np.isfinite(value)) for value in (left, right, lam, gate_left, gate_right, x, s, yh)):
        raise ValueError("all inputs must be finite")
    if np.any(lam <= 0.0) or s[0] <= 0.0 or np.any(s <= 1.0e-6 * s[0]):
        raise ValueError("all four local singular modes must be active")
    if np.any(s[:-1] - s[1:] <= 1.0e-10 * s[0]):
        raise ValueError("local singular values must be nondegenerate")
    if np.linalg.norm(x.conj().T @ x - np.eye(4)) > 1.0e-10 or np.linalg.norm(yh @ yh.conj().T - np.eye(4)) > 1.0e-10:
        raise ValueError("singular-vector matrices must be unitary")

    yh_tensor = yh.reshape(4, 2, 2)
    left_basis = np.einsum("bqt,tcd,kqd->bck", gate_right, right, yh_tensor.conj())
    x_tensor = x.reshape(2, 2, 4)
    right_numerator = np.einsum("aok,bos,a,sac->bkc", x_tensor.conj(), gate_left, lam, left)
    right_basis = right_numerator / s[None, :, None]

    lhs_left = np.einsum("bos,sac,bck->oak", gate_left, left, left_basis)
    target_left = ((x_tensor * s[None, None, :]) / lam[:, None, None]).transpose(1, 0, 2)
    lhs_right = np.einsum("bkc,bqt,tcd->kqd", right_basis, gate_right, right)
    if np.linalg.norm(lhs_left - target_left) > 1.0e-10 or np.linalg.norm(lhs_right - yh_tensor) > 1.0e-10:
        raise ValueError("projector bases do not reproduce the local SVD factors")
    return np.asarray(left_basis, dtype=np.complex128), np.asarray(right_basis, dtype=np.complex128)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    # Evaluate each side lazily with its own implementation chain and fresh inputs.
    setup_1 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases):
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
                    z = 11 * (n + 1) + 7 * (sig + 1) + 5 * (a + 1) + 3 * (b + 1) + (n + 1) * (sig + 1) * (a + b + 2)
                    t[sig, a, b] = np.sin(0.37 * z) + 0.31 * np.cos(0.19 * (z * z + 3 * a + 5 * b)) + 1j * (np.cos(0.29 * (z + 2 * a * b)) + 0.23 * np.sin(0.41 * (z * z + sig + b)))
        raw.append(t)
    canonical, spectra = canonicalize_initial_mps(tuple(raw))
    gl, gr = build_operator_schmidt_factors(np.array([0]), np.array([0.31]))
    left_tensor = canonical[1]
    right_tensor = canonical[2]
    left_bond_singular_values = spectra[0]
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]
    C, Theta, left_singular_vectors, singular_values, right_singular_vectors_h = compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors)
    return _pack(construct_tebd_projector_bases(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors, left_singular_vectors, singular_values, right_singular_vectors_h))
'''
    setup_2 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases):
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
                    z = 11 * (n + 1) + 7 * (sig + 1) + 5 * (a + 1) + 3 * (b + 1) + (n + 1) * (sig + 1) * (a + b + 2)
                    t[sig, a, b] = np.sin(0.37 * z) + 0.31 * np.cos(0.19 * (z * z + 3 * a + 5 * b)) + 1j * (np.cos(0.29 * (z + 2 * a * b)) + 0.23 * np.sin(0.41 * (z * z + sig + b)))
        raw.append(t)
    canonical, spectra = canonicalize_initial_mps(tuple(raw))
    gl, gr = build_operator_schmidt_factors(np.array([1]), np.array([0.27]))
    left_tensor = canonical[3]
    right_tensor = canonical[4]
    left_bond_singular_values = spectra[2]
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]
    C, Theta, left_singular_vectors, singular_values, right_singular_vectors_h = compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors)
    return _pack(construct_tebd_projector_bases(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors, left_singular_vectors, singular_values, right_singular_vectors_h))
'''
    setup_3 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases):
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
                    z = 11 * (n + 1) + 7 * (sig + 1) + 5 * (a + 1) + 3 * (b + 1) + (n + 1) * (sig + 1) * (a + b + 2)
                    t[sig, a, b] = np.sin(0.37 * z) + 0.31 * np.cos(0.19 * (z * z + 3 * a + 5 * b)) + 1j * (np.cos(0.29 * (z + 2 * a * b)) + 0.23 * np.sin(0.41 * (z * z + sig + b)))
        raw.append(t)
    canonical, spectra = canonicalize_initial_mps(tuple(raw))
    gl, gr = build_operator_schmidt_factors(np.array([2]), np.array([0.63]))
    left_tensor = canonical[2]
    right_tensor = canonical[3]
    left_bond_singular_values = spectra[1]
    left_gate_factors = gl[0]
    right_gate_factors = gr[0]
    C, Theta, left_singular_vectors, singular_values, right_singular_vectors_h = compute_local_tebd_svd(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors)
    return _pack(construct_tebd_projector_bases(left_tensor, right_tensor, left_bond_singular_values, left_gate_factors, right_gate_factors, left_singular_vectors, singular_values, right_singular_vectors_h))
'''
    return [
        {
            "setup": setup_1,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_compute_local_tebd_svd, _oracle_construct_tebd_projector_bases)',
            "tol": 2e-10,
        },
        {
            "setup": setup_2,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_compute_local_tebd_svd, _oracle_construct_tebd_projector_bases)',
            "tol": 2e-10,
        },
        {
            "setup": setup_3,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, compute_local_tebd_svd, construct_tebd_projector_bases)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_compute_local_tebd_svd, _oracle_construct_tebd_projector_bases)',
            "tol": 2e-10,
        },
    ]

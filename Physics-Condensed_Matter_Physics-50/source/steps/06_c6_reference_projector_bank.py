"""
Build the complete time-ordered TEBD projector bank along the deterministic rank-two reference trajectory.

At each prescribed internal bond, the current right-canonical reference tensors generate the complete local SVD and the \(Q_L^b,Q_R^b\) projector bases.  The selection marginals are computed from all size-two mode subsets.  The reference trajectory itself retains the two leading singular modes without stochastic weighting, updates the two local canonical tensors, and carries the new two-entry bond spectrum into later overlapping events.

Returns
-------
Return complex128 left/right banks of shapes `(L,2,2,4)` and `(L,2,4,2)`, float64 complete spectra `(L,4)`, and float64 marginal inclusion probabilities `(L,4)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_reference_projector_bank(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_singular_values: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return time-ordered projector bases, spectra, and inclusion marginals.

    For ``L`` gate events the outputs have shapes ``(L,2,2,4)``,
    ``(L,2,4,2)``, ``(L,4)``, and ``(L,4)``.  Scientific bond indices are
    one-based and must select internal rank-two bonds.
    """
    return left_projector_bank, right_projector_bank, local_spectra, inclusion_probabilities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_reference_projector_bank(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_singular_values: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    if not isinstance(canonical_tensors, tuple) or len(canonical_tensors) != 6:
        raise ValueError("six canonical tensors are required")
    if not isinstance(bond_singular_values, tuple) or len(bond_singular_values) != 5:
        raise ValueError("five bond spectra are required")
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in canonical_tensors]
    spectra = [np.asarray(v, dtype=np.float64).copy() for v in bond_singular_values]
    bonds = np.asarray(bond_indices)
    gate_left = np.asarray(left_gate_factors, dtype=np.complex128)
    gate_right = np.asarray(right_gate_factors, dtype=np.complex128)
    if retained_dimension != 2:
        raise ValueError("the retained dimension is fixed at two")
    if bonds.ndim != 1 or bonds.size == 0 or bonds.dtype.kind not in "iu":
        raise ValueError("bond_indices must be a nonempty integer vector")
    if np.any((bonds < 2) | (bonds > 4)):
        raise ValueError("events must act on internal bonds 2, 3, or 4")
    event_count = bonds.size
    if gate_left.shape != (event_count, 2, 2, 2) or gate_right.shape != gate_left.shape:
        raise ValueError("gate-factor arrays do not match the event count")
    if any(not np.all(np.isfinite(value)) for value in tensors + spectra + [gate_left, gate_right]):
        raise ValueError("all inputs must be finite")
    left_bank = np.empty((event_count, 2, 2, 4), dtype=np.complex128)
    right_bank = np.empty((event_count, 2, 4, 2), dtype=np.complex128)
    local_spectra = np.empty((event_count, 4), dtype=np.float64)
    marginals = np.empty((event_count, 4), dtype=np.float64)

    for event, scientific_bond in enumerate(bonds.astype(np.int64)):
        left_site = scientific_bond - 1
        lam_left = spectra[scientific_bond - 2]
        _, _, x, s, yh = _oracle_compute_local_tebd_svd(
            tensors[left_site], tensors[left_site + 1], lam_left,
            gate_left[event], gate_right[event]
        )
        ql, qr = _oracle_construct_tebd_projector_bases(
            tensors[left_site], tensors[left_site + 1], lam_left,
            gate_left[event], gate_right[event], x, s, yh
        )
        _, _, inclusion, _ = _oracle_compute_subspace_statistics(s, retained_dimension)
        left_bank[event] = ql
        right_bank[event] = qr
        local_spectra[event] = s
        marginals[event] = inclusion

        x_tensor = x[:, :retained_dimension].reshape(2, 2, retained_dimension)
        tensors[left_site] = (
            x_tensor * s[None, None, :retained_dimension] / lam_left[:, None, None]
        ).transpose(1, 0, 2)
        tensors[left_site + 1] = yh[:retained_dimension].reshape(retained_dimension, 2, 2).transpose(1, 0, 2)
        spectra[scientific_bond - 1] = s[:retained_dimension].copy()
    return left_bank, right_bank, local_spectra, marginals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    # Evaluate each side lazily with its own implementation chain and fresh inputs.
    setup_1 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank):
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
    canonical_tensors, bond_singular_values = canonicalize_initial_mps(tuple(raw))
    retained_dimension = 2
    bond_indices = np.array([2, 4, 3, 2, 4, 3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([0, 1, 2, 1, 2, 0]), np.array([0.31, 0.27, 0.23, 0.41, 0.35, 0.29]))
    return _pack(build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, retained_dimension))
'''
    setup_2 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank):
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
    canonical_tensors, bond_singular_values = canonicalize_initial_mps(tuple(raw))
    retained_dimension = 2
    bond_indices = np.array([3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([2]), np.array([0.52]))
    return _pack(build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, retained_dimension))
'''
    setup_3 = '''def _evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank):
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
    canonical_tensors, bond_singular_values = canonicalize_initial_mps(tuple(raw))
    retained_dimension = 2
    bond_indices = np.array([4, 2, 3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([1, 0, 2]), np.array([0.18, 0.44, 0.33]))
    return _pack(build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, retained_dimension))
'''
    return [
        {
            "setup": setup_1,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_build_reference_projector_bank)',
            "tol": 3e-10,
        },
        {
            "setup": setup_2,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_build_reference_projector_bank)',
            "tol": 3e-10,
        },
        {
            "setup": setup_3,
            "call": '_evaluate(build_operator_schmidt_factors, canonicalize_initial_mps, build_reference_projector_bank)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_canonicalize_initial_mps, _oracle_build_reference_projector_bank)',
            "tol": 3e-10,
        },
    ]

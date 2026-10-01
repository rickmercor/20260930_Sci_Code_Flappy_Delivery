"""
Propagate fixed stochastic subspace histories through the stored TEBD projector bank without intermediate normalization.

Each history row supplies an increasing pair of selected mode indices for every event.  The left selector column for mode (k) is weighted by the reciprocal marginal inclusion probability \(1/r_k\), while the right selector is unweighted.  The resulting \(P_L^b=Q_L^bS_w\) and \(P_R^b=S^TQ_R^b\) act through the operator-Schmidt gate factors on the current neighboring MPS tensors; all accumulated complex amplitudes are retained.

Returns
-------
Return a complex128 array of shape `(M,64)` containing the unnormalized dense state for every fixed history, with input history order preserved.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_weighted_histories(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_projector_bank: "np.ndarray",
    right_projector_bank: "np.ndarray",
    inclusion_probabilities: "np.ndarray",
    histories: "np.ndarray",
) -> "np.ndarray":
    """Return the unnormalized dense states for all fixed histories.

    ``histories`` has shape ``(M,L,2)`` and contains strictly increasing
    zero-based mode pairs.  The return is a complex array of shape ``(M,64)``;
    no trajectory is normalized during or after propagation.  Raise
    ``ValueError`` for invalid shapes, indices, ordering, or probabilities.
    """
    return trajectory_states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_propagate_weighted_histories(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_projector_bank: "np.ndarray",
    right_projector_bank: "np.ndarray",
    inclusion_probabilities: "np.ndarray",
    histories: "np.ndarray",
) -> "np.ndarray":
    if not isinstance(canonical_tensors, tuple) or len(canonical_tensors) != 6:
        raise ValueError("six canonical tensors are required")
    initial = tuple(np.asarray(t, dtype=np.complex128) for t in canonical_tensors)
    bonds = np.asarray(bond_indices)
    gate_left = np.asarray(left_gate_factors, dtype=np.complex128)
    gate_right = np.asarray(right_gate_factors, dtype=np.complex128)
    ql_bank = np.asarray(left_projector_bank, dtype=np.complex128)
    qr_bank = np.asarray(right_projector_bank, dtype=np.complex128)
    inclusion = np.asarray(inclusion_probabilities, dtype=np.float64)
    selections = np.asarray(histories)
    if bonds.ndim != 1 or bonds.size == 0 or bonds.dtype.kind not in "iu":
        raise ValueError("bond_indices must be a nonempty integer vector")
    event_count = bonds.size
    if np.any((bonds < 2) | (bonds > 4)):
        raise ValueError("events must act on internal bonds 2, 3, or 4")
    if gate_left.shape != (event_count, 2, 2, 2) or gate_right.shape != gate_left.shape:
        raise ValueError("gate factors do not match the event count")
    if ql_bank.shape != (event_count, 2, 2, 4) or qr_bank.shape != (event_count, 2, 4, 2):
        raise ValueError("projector-bank shapes are invalid")
    if inclusion.shape != (event_count, 4):
        raise ValueError("inclusion probabilities must have shape (L,4)")
    if selections.ndim != 3 or selections.shape[1:] != (event_count, 2) or selections.shape[0] == 0:
        raise ValueError("histories must have shape (M,L,2)")
    if selections.dtype.kind not in "iu" or np.any((selections < 0) | (selections > 3)):
        raise ValueError("history mode indices must be integers from zero through three")
    if np.any(selections[:, :, 0] >= selections[:, :, 1]):
        raise ValueError("each selected mode pair must be strictly increasing")
    if any(not np.all(np.isfinite(value)) for value in initial + (gate_left, gate_right, ql_bank, qr_bank, inclusion)):
        raise ValueError("all numerical inputs must be finite")
    if np.any(inclusion <= 0.0) or np.any(inclusion > 1.0 + 1.0e-12):
        raise ValueError("all selectable modes need positive inclusion probabilities")

    states = np.empty((selections.shape[0], 64), dtype=np.complex128)
    for sample in range(selections.shape[0]):
        tensors = [tensor.copy() for tensor in initial]
        for event, scientific_bond in enumerate(bonds.astype(np.int64)):
            left_site = scientific_bond - 1
            chosen = selections[sample, event].astype(np.int64)
            weighted_left = np.take(ql_bank[event], chosen, axis=2)
            weighted_left = weighted_left / inclusion[event, chosen][None, None, :]
            selected_right = np.take(qr_bank[event], chosen, axis=1)
            new_left = np.einsum(
                "bos,sac,bcn->oan", gate_left[event], tensors[left_site], weighted_left
            )
            new_right = np.einsum(
                "bqt,bnc,tcd->qnd", gate_right[event], selected_right, tensors[left_site + 1]
            )
            tensors[left_site] = np.asarray(new_left, dtype=np.complex128)
            tensors[left_site + 1] = np.asarray(new_right, dtype=np.complex128)
        states[sample] = _contract_six_site_mps(tuple(tensors))
    return states

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    # Evaluate each side lazily with its own implementation chain and fresh inputs.
    setup_1 = '''def _evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories):
    import numpy as np
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
    bond_indices = np.array([2, 4, 3, 2, 4, 3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([0, 1, 2, 1, 2, 0]), np.array([0.31, 0.27, 0.23, 0.41, 0.35, 0.29]))
    left_projector_bank, right_projector_bank, local_spectra, inclusion_probabilities = build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, 2)
    histories = np.array([[[0, 1], [0, 2], [1, 2], [0, 1], [0, 3], [1, 3]], [[1, 3], [0, 1], [0, 2], [1, 2], [0, 1], [0, 3]]], dtype=np.int64)
    return propagate_weighted_histories(canonical_tensors, bond_indices, left_gate_factors, right_gate_factors, left_projector_bank, right_projector_bank, inclusion_probabilities, histories)
'''
    setup_2 = '''def _evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories):
    import numpy as np
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
    bond_indices = np.array([2, 4, 3, 2, 4, 3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([0, 1, 2, 1, 2, 0]), np.array([0.31, 0.27, 0.23, 0.41, 0.35, 0.29]))
    left_projector_bank, right_projector_bank, local_spectra, inclusion_probabilities = build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, 2)
    histories = np.array([[[0, 3], [1, 3], [0, 1], [0, 2], [1, 2], [0, 1]], [[0, 2], [0, 3], [0, 1], [0, 2], [0, 3], [0, 1]], [[1, 2], [1, 3], [0, 1], [1, 2], [1, 3], [0, 1]]], dtype=np.int64)
    return propagate_weighted_histories(canonical_tensors, bond_indices, left_gate_factors, right_gate_factors, left_projector_bank, right_projector_bank, inclusion_probabilities, histories)
'''
    setup_3 = '''def _evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories):
    import numpy as np
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
    bond_indices = np.array([2, 4, 3, 2, 4, 3], dtype=np.int64)
    left_gate_factors, right_gate_factors = build_operator_schmidt_factors(np.array([0, 1, 2, 1, 2, 0]), np.array([0.31, 0.27, 0.23, 0.41, 0.35, 0.29]))
    left_projector_bank, right_projector_bank, local_spectra, inclusion_probabilities = build_reference_projector_bank(canonical_tensors, bond_singular_values, bond_indices, left_gate_factors, right_gate_factors, 2)
    histories = np.array([[[0, 0], [0, 2], [1, 2], [0, 1], [0, 3], [1, 3]]], dtype=np.int64)

    def candidate_result():
        try:
            propagate_weighted_histories(canonical_tensors, bond_indices, left_gate_factors, right_gate_factors, left_projector_bank, right_projector_bank, inclusion_probabilities, histories)
            return 0
        except ValueError:
            return 1
    return candidate_result()
'''
    return [
        {
            "setup": setup_1,
            "call": '_evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_build_reference_projector_bank, _oracle_canonicalize_initial_mps, _oracle_propagate_weighted_histories)',
            "tol": 2e-09,
        },
        {
            "setup": setup_2,
            "call": '_evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_build_reference_projector_bank, _oracle_canonicalize_initial_mps, _oracle_propagate_weighted_histories)',
            "tol": 2e-09,
        },
        {
            "setup": setup_3,
            "call": '_evaluate(build_operator_schmidt_factors, build_reference_projector_bank, canonicalize_initial_mps, propagate_weighted_histories)',
            "gold_call": '_evaluate(_oracle_build_operator_schmidt_factors, _oracle_build_reference_projector_bank, _oracle_canonicalize_initial_mps, _oracle_propagate_weighted_histories)',
        },
    ]

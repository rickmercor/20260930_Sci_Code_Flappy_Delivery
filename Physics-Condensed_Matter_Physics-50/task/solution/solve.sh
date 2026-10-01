#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
import scipy.linalg


def _phase_fix_right_rows(
    left_vectors: "np.ndarray",
    right_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    left = np.asarray(left_vectors, dtype=np.complex128).copy()
    right_h = np.asarray(right_vectors_h, dtype=np.complex128).copy()
    for k in range(right_h.shape[0]):
        magnitudes = np.abs(right_h[k])
        pivot = int(np.argmax(magnitudes))
        if magnitudes[pivot] == 0.0:
            raise ValueError("a singular vector has no nonzero pivot")
        phase = right_h[k, pivot] / magnitudes[pivot]
        right_h[k] /= phase
        left[:, k] *= phase
    return left, right_h


def _phase_fix_left_columns(
    left_vectors: "np.ndarray",
    right_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    left = np.asarray(left_vectors, dtype=np.complex128).copy()
    right_h = np.asarray(right_vectors_h, dtype=np.complex128).copy()
    for k in range(left.shape[1]):
        magnitudes = np.abs(left[:, k])
        pivot = int(np.argmax(magnitudes))
        if magnitudes[pivot] == 0.0:
            raise ValueError("a singular vector has no nonzero pivot")
        phase = left[pivot, k] / magnitudes[pivot]
        left[:, k] /= phase
        right_h[k] *= phase
    return left, right_h


def _validated_six_site_tensors(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[np.ndarray, ...]":
    if not isinstance(site_tensors, tuple) or len(site_tensors) != 6:
        raise ValueError("site_tensors must be a tuple of six tensors")
    tensors = tuple(np.asarray(tensor, dtype=np.complex128) for tensor in site_tensors)
    if any(tensor.ndim != 3 or tensor.shape[0] != 2 for tensor in tensors):
        raise ValueError("every tensor must have shape (2, chi_left, chi_right)")
    if any(not np.all(np.isfinite(tensor)) for tensor in tensors):
        raise ValueError("site tensors must be finite")
    if tensors[0].shape[1] != 1 or tensors[-1].shape[2] != 1:
        raise ValueError("outer MPS bond dimensions must be one")
    if any(tensors[n].shape[2] != tensors[n + 1].shape[1] for n in range(5)):
        raise ValueError("adjacent virtual bond dimensions are incompatible")
    return tensors


def _contract_six_site_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "np.ndarray":
    tensors = _validated_six_site_tensors(site_tensors)
    block = tensors[0][:, 0, :]
    for tensor in tensors[1:]:
        block = np.tensordot(block, tensor, axes=([-1], [1]))
    return np.asarray(block[..., 0], dtype=np.complex128).reshape(64)


def canonicalize_initial_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]":
    tensors = _validated_six_site_tensors(site_tensors)
    state = _contract_six_site_mps(tensors)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("the contracted state must be finite and nonzero")
    work = (state / norm).reshape(32, 2)
    canonical = [None] * 6
    spectra = [None] * 5
    right_dimension = 1

    for site in range(5, 0, -1):
        matrix = work.reshape(2 ** site, 2 * right_dimension)
        left, singular_values, right_h = scipy.linalg.svd(
            matrix, full_matrices=False, lapack_driver="gesvd"
        )
        if singular_values.size < 2 or singular_values[1] <= 1.0e-12 * singular_values[0]:
            raise ValueError("each internal cut must have Schmidt rank exactly two")
        if singular_values.size > 2 and singular_values[2] > 1.0e-12 * singular_values[0]:
            raise ValueError("each internal cut must have Schmidt rank exactly two")
        active = singular_values[:2]
        if active[0] - active[1] <= 1.0e-12 * active[0]:
            raise ValueError("initial active Schmidt values must be nondegenerate")
        left, right_h = _phase_fix_right_rows(left[:, :2], right_h[:2])
        canonical[site] = right_h.reshape(2, 2, right_dimension).transpose(1, 0, 2)
        spectra[site - 1] = np.asarray(active, dtype=np.float64)
        work = left * active[np.newaxis, :]
        right_dimension = 2

    canonical[0] = np.asarray(work.reshape(2, 1, 2), dtype=np.complex128)
    result_tensors = tuple(np.asarray(tensor, dtype=np.complex128) for tensor in canonical)
    result_spectra = tuple(np.asarray(values, dtype=np.float64) for values in spectra)
    return result_tensors, result_spectra

import numpy as np


def build_operator_schmidt_factors(
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

import numpy as np
import scipy.linalg


def compute_local_tebd_svd(
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

import numpy as np


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

import itertools
import numpy as np


def compute_subspace_statistics(
    singular_values: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    s = np.asarray(singular_values, dtype=np.float64)
    if s.shape != (4,) or not np.all(np.isfinite(s)):
        raise ValueError("singular_values must be a finite vector of length four")
    if retained_dimension != 2:
        raise ValueError("the retained dimension is fixed at two")
    if s[0] <= 0.0 or np.any(s < 0.0) or np.any(s[:-1] < s[1:]):
        raise ValueError("singular values must be nonnegative and nonincreasing")
    weights = np.maximum(s, 1.0e-12 * s[0])
    subsets = np.asarray(list(itertools.combinations(range(4), 2)), dtype=np.int64)
    products = np.prod(weights[subsets], axis=1)
    partition = float(np.sum(products))
    if not np.isfinite(partition) or partition <= 0.0:
        raise ValueError("subset weights must have positive finite normalization")
    joint = products / partition
    marginal = np.zeros(4, dtype=np.float64)
    for probability, subset in zip(joint, subsets):
        marginal[subset] += probability

    suffix = np.zeros((5, 3), dtype=np.float64)
    suffix[4, 2] = 1.0
    for k in range(3, -1, -1):
        for m in range(2, -1, -1):
            skip = suffix[k + 1, m]
            take = weights[k] * suffix[k + 1, m + 1] if m < 2 else 0.0
            suffix[k, m] = skip + take
    conditional = np.zeros((4, 2), dtype=np.float64)
    for k in range(4):
        for m in range(2):
            take = weights[k] * suffix[k + 1, m + 1]
            skip = suffix[k + 1, m]
            denom = take + skip
            conditional[k, m] = take / denom if denom > 0.0 else 0.0
    return subsets, joint, marginal, conditional

import numpy as np


def build_reference_projector_bank(
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
        _, _, x, s, yh = compute_local_tebd_svd(
            tensors[left_site], tensors[left_site + 1], lam_left,
            gate_left[event], gate_right[event]
        )
        ql, qr = construct_tebd_projector_bases(
            tensors[left_site], tensors[left_site + 1], lam_left,
            gate_left[event], gate_right[event], x, s, yh
        )
        _, _, inclusion, _ = compute_subspace_statistics(s, retained_dimension)
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

import numpy as np


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

import numpy as np


def compute_direct_cross_estimator(
    bra_states: "np.ndarray",
    ket_states: "np.ndarray",
    observable: "np.ndarray",
) -> "tuple[np.ndarray, complex]":
    bra = np.asarray(bra_states, dtype=np.complex128)
    ket = np.asarray(ket_states, dtype=np.complex128)
    operator = np.asarray(observable, dtype=np.complex128)
    if bra.ndim != 2 or bra.shape[1] != 64 or ket.shape != bra.shape or bra.shape[0] == 0:
        raise ValueError("bra and ket ensembles must have matching nonempty shape (M,64)")
    if operator.shape != (64, 64):
        raise ValueError("observable must have shape (64,64)")
    if any(not np.all(np.isfinite(value)) for value in (bra, ket, operator)):
        raise ValueError("all inputs must be finite")
    if np.linalg.norm(operator - operator.conj().T) > 1.0e-12:
        raise ValueError("observable must be Hermitian")
    samples = np.einsum("mi,ij,mj->m", bra.conj(), operator, ket)
    mean = complex(np.mean(samples, dtype=np.complex128))
    return np.asarray(samples, dtype=np.complex128), mean

import numpy as np


def compute_final_estimate(
    site_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
    ket_histories: "np.ndarray",
    bra_histories: "np.ndarray",
    observable: "np.ndarray",
) -> float:
    ket_choices = np.asarray(ket_histories)
    bra_choices = np.asarray(bra_histories)
    if ket_choices.shape != bra_choices.shape or ket_choices.ndim != 3 or ket_choices.shape[0] == 0:
        raise ValueError("ket and bra histories must have matching nonempty shape (M,L,2)")
    canonical, spectra = canonicalize_initial_mps(site_tensors)
    gate_left, gate_right = build_operator_schmidt_factors(pauli_axes, angles)
    ql_bank, qr_bank, _, inclusion = build_reference_projector_bank(
        canonical, spectra, bond_indices, gate_left, gate_right, 2
    )
    ket_states = propagate_weighted_histories(
        canonical, bond_indices, gate_left, gate_right, ql_bank, qr_bank, inclusion, ket_choices
    )
    bra_states = propagate_weighted_histories(
        canonical, bond_indices, gate_left, gate_right, ql_bank, qr_bank, inclusion, bra_choices
    )
    _, direct_mean = compute_direct_cross_estimator(bra_states, ket_states, observable)
    return float(np.round(direct_mean.real, 10))
SCICODE_GOLD_EOF

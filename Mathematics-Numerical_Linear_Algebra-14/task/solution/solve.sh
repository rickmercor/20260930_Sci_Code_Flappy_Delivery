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


def compute_skew_polar_factor(
    A: np.ndarray,
    skew_tolerance: float = 1e-12,
    singularity_tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference polar-factor calculation."""
    raw = np.asarray(A)
    if np.iscomplexobj(raw):
        raise ValueError("A must be real")
    matrix = np.asarray(raw, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or matrix.shape[0] % 2
    ):
        raise ValueError("A must be a nonempty even-order square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite entries")
    if not np.isfinite(skew_tolerance) or skew_tolerance <= 0.0:
        raise ValueError("skew_tolerance must be positive and finite")
    if not np.isfinite(singularity_tolerance) or singularity_tolerance <= 0.0:
        raise ValueError("singularity_tolerance must be positive and finite")

    scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    defect = float(np.linalg.norm(matrix + matrix.T, ord="fro") / scale)
    if defect > skew_tolerance:
        raise ValueError("A must be skew-symmetric within skew_tolerance")

    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[-1] <= singularity_tolerance * singular_values[0]:
        raise ValueError("A is numerically singular")
    return left @ right_t

import numpy as np


def sample_positive_imaginary_basis(P: np.ndarray, seed: int) -> np.ndarray:
    """Reference seeded invariant-subspace basis with its QR diagonal moduli."""
    raw = np.asarray(P)
    if np.iscomplexobj(raw):
        raise ValueError("P must be real")
    polar = np.asarray(raw, dtype=float)
    if (
        polar.ndim != 2
        or polar.shape[0] != polar.shape[1]
        or polar.shape[0] < 2
        or polar.shape[0] % 2
    ):
        raise ValueError("P must be an even-order square matrix")
    if not np.all(np.isfinite(polar)):
        raise ValueError("P must contain only finite entries")
    identity = np.eye(polar.shape[0])
    if not np.allclose(polar + polar.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("P must be skew-symmetric")
    if not np.allclose(polar.T @ polar, identity, atol=1e-10, rtol=0.0):
        raise ValueError("P must be orthogonal")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n = polar.shape[0] // 2
    sample = np.random.default_rng(int(seed)).standard_normal((2 * n, n))
    basis, triangular = np.linalg.qr(
        (polar + 1j * identity) @ sample,
        mode="reduced",
    )
    diagonal = np.diag(triangular)
    moduli = np.abs(diagonal)
    if np.any(moduli <= 100.0 * np.finfo(float).eps):
        raise ValueError("the sampled range is numerically rank deficient")
    phases = diagonal / moduli
    return np.vstack((basis * phases, moduli.astype(complex)))

import numpy as np


def build_real_symplectic_basis(V_tilde: np.ndarray) -> np.ndarray:
    """Reference complex-to-real basis map."""
    basis = np.asarray(V_tilde, dtype=complex)
    if basis.ndim != 2 or basis.shape[0] != 2 * basis.shape[1] or basis.shape[1] < 1:
        raise ValueError("V_tilde must have shape (2n, n)")
    if not np.all(np.isfinite(basis.real)) or not np.all(np.isfinite(basis.imag)):
        raise ValueError("V_tilde must contain only finite entries")
    identity = np.eye(basis.shape[1])
    if not np.allclose(basis.conj().T @ basis, identity, atol=1e-10, rtol=0.0):
        raise ValueError("V_tilde must have orthonormal columns")
    return np.sqrt(2.0) * np.concatenate((basis.real, -basis.imag), axis=1)

import numpy as np


def compute_skew_hamiltonian_blocks(
    A: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Reference block compression."""
    matrix = np.asarray(A, dtype=float)
    basis = np.asarray(Z, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or matrix.shape[0] % 2
    ):
        raise ValueError("A must be an even-order square matrix")
    if basis.shape != matrix.shape:
        raise ValueError("Z must have the same shape as A")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(basis)):
        raise ValueError("A and Z must contain only finite entries")
    if not np.allclose(matrix + matrix.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("A must be skew-symmetric")
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[0]),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("Z must be orthogonal")

    n = matrix.shape[0] // 2
    first = basis[:, :n]
    second = basis[:, n:]
    omega = first.T @ matrix @ first
    hessian = second.T @ matrix @ first
    omega = 0.5 * (omega - omega.T)
    hessian = 0.5 * (hessian + hessian.T)
    return np.stack((hessian, omega))

import numpy as np


def _nla14_projector_pivot_gauge(
    cluster_vectors: np.ndarray,
    negative_orientation: bool,
    balanced: bool = False,
) -> np.ndarray:
    """Choose a deterministic basis from a Hermitian spectral projector."""
    projector = cluster_vectors @ cluster_vectors.conj().T
    projector = 0.5 * (projector + projector.conj().T)
    order, multiplicity = cluster_vectors.shape
    accepted = np.empty((order, 0), dtype=complex)
    unused = list(range(order))
    pivots = []
    floor = 100.0 * np.finfo(float).eps

    for _ in range(multiplicity):
        residuals = []
        scores = np.empty(len(unused), dtype=float)
        for location, coordinate in enumerate(unused):
            residual = projector[:, coordinate].copy()
            for _pass in range(2):
                if accepted.shape[1]:
                    residual -= accepted @ (accepted.conj().T @ residual)
            residuals.append(residual)
            scores[location] = np.linalg.norm(residual)

        best = float(scores.max())
        tied = np.flatnonzero(scores >= best * (1.0 - 1e-12))
        location = int(tied[-1] if negative_orientation else tied[0])
        coordinate = unused.pop(location)
        pivots.append(coordinate)
        residual = residuals[location]
        norm = float(np.linalg.norm(residual))
        if norm <= floor:
            raise ValueError("a repeated eigenspace has no stable residual pivot")

        vector = residual / norm
        pivot_value = vector[coordinate]
        if abs(pivot_value) <= floor:
            raise ValueError("a repeated eigenspace has no stable phase pivot")
        vector /= pivot_value / abs(pivot_value)
        accepted = np.column_stack((accepted, vector))

    if balanced:
        columns = projector[:, pivots]
        gram = columns.conj().T @ columns
        gram = 0.5 * (gram + gram.conj().T)
        gram_values, gram_vectors = np.linalg.eigh(gram)
        if float(gram_values[0]) <= floor**2:
            raise ValueError("a tight eigenspace has no stable inverse square root")
        inverse_root = (gram_vectors / np.sqrt(gram_values)) @ gram_vectors.conj().T
        accepted = columns @ inverse_root
        for column, coordinate in enumerate(pivots):
            pivot_value = accepted[coordinate, column]
            if abs(pivot_value) <= floor:
                raise ValueError("a tight eigenspace has no stable phase pivot")
            accepted[:, column] /= pivot_value / abs(pivot_value)

    return accepted


def solve_half_size_hermitian_problem(
    blocks: np.ndarray,
) -> np.ndarray:
    """Reference Hermitian eigensolve with deterministic phases."""
    pair = np.asarray(blocks, dtype=float)
    if (
        pair.ndim != 3
        or pair.shape[0] != 2
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] < 1
    ):
        raise ValueError("blocks must have shape (2, n, n)")
    if not np.all(np.isfinite(pair)):
        raise ValueError("blocks must contain only finite entries")
    hessian, omega = pair
    if not np.allclose(hessian, hessian.T, atol=1e-10, rtol=0.0):
        raise ValueError("blocks[0] must be symmetric")
    if not np.allclose(omega + omega.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("blocks[1] must be skew-symmetric")

    hermitian = hessian + 1j * omega
    values, vectors = np.linalg.eigh(hermitian)
    cluster_tolerance = (
        256.0
        * np.finfo(float).eps
        * max(1.0, float(np.linalg.norm(hermitian, ord=2)))
    )
    exact_tolerance = cluster_tolerance / 8.0

    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[stop] - values[start] <= cluster_tolerance:
            stop += 1

        if stop - start == 1:
            magnitudes = np.abs(vectors[:, start])
            tied = np.flatnonzero(
                magnitudes >= magnitudes.max() * (1.0 - 1e-12)
            )
            pivot = int(tied[-1] if values[start] < 0.0 else tied[0])
            pivot_value = vectors[pivot, start]
            if abs(pivot_value) <= 100.0 * np.finfo(float).eps:
                raise ValueError("an eigenvector has no stable phase pivot")
            vectors[:, start] /= pivot_value / abs(pivot_value)
        else:
            representative = float(np.mean(values[start:stop]))
            span = float(values[stop - 1] - values[start])
            vectors[:, start:stop] = _nla14_projector_pivot_gauge(
                vectors[:, start:stop],
                negative_orientation=representative < 0.0,
                balanced=span > exact_tolerance,
            )
            values[start:stop] = representative
        start = stop

    order = np.argsort(-np.abs(values), kind="stable")
    ordered_values = values[order]
    ordered_vectors = vectors[:, order]
    return np.vstack((ordered_values.astype(complex), ordered_vectors))

import numpy as np


def recover_real_spectral_decomposition(
    Z: np.ndarray,
    eigensystem: np.ndarray,
) -> np.ndarray:
    """Reference signed real recovery."""
    basis = np.asarray(Z, dtype=float)
    packed = np.asarray(eigensystem, dtype=complex)
    if (
        basis.ndim != 2
        or basis.shape[0] != basis.shape[1]
        or basis.shape[0] < 2
        or basis.shape[0] % 2
    ):
        raise ValueError("Z must be an even-order square matrix")
    n = basis.shape[0] // 2
    if packed.shape != (n + 1, n):
        raise ValueError("eigensystem must have shape (n + 1, n)")
    if not np.allclose(packed[0].imag, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("the packed eigenvalues must be real")
    values = packed[0].real
    vectors = packed[1:]
    if (
        not np.all(np.isfinite(basis))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(vectors.real))
        or not np.all(np.isfinite(vectors.imag))
    ):
        raise ValueError("all inputs must contain only finite entries")
    if not np.allclose(basis.T @ basis, np.eye(2 * n), atol=1e-10, rtol=0.0):
        raise ValueError("Z must be orthogonal")
    if not np.allclose(vectors.conj().T @ vectors, np.eye(n), atol=1e-10, rtol=0.0):
        raise ValueError("U must be unitary")

    first = basis[:, :n]
    second = basis[:, n:]
    real = vectors.real
    imaginary = vectors.imag
    signs = np.where(values < 0.0, -1.0, 1.0)
    q_first = first @ imaginary + second @ real
    q_second = (-first @ real + second @ imaginary) * signs
    spectral_basis = np.concatenate((q_first, q_second), axis=1)
    result = np.zeros((2 * n + 1, 2 * n), dtype=float)
    result[: 2 * n] = spectral_basis
    result[2 * n, :n] = np.abs(values)
    return result

import numpy as np


def compute_basis_checksums(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Reference weighted basis checksums."""
    first_basis = np.asarray(Z, dtype=float)
    spectral_basis = np.asarray(Q, dtype=float)
    if (
        first_basis.ndim != 2
        or first_basis.shape[0] != first_basis.shape[1]
        or first_basis.shape[0] < 2
        or first_basis.shape[0] % 2
    ):
        raise ValueError("Z must be an even-order square matrix")
    if spectral_basis.shape != first_basis.shape:
        raise ValueError("Q must have the same shape as Z")
    if not np.all(np.isfinite(first_basis)) or not np.all(np.isfinite(spectral_basis)):
        raise ValueError("Z and Q must contain only finite entries")
    identity = np.eye(first_basis.shape[0])
    if not np.allclose(first_basis.T @ first_basis, identity, atol=1e-10, rtol=0.0):
        raise ValueError("Z must be orthogonal")
    if not np.allclose(
        spectral_basis.T @ spectral_basis,
        identity,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("Q must be orthogonal")

    order = first_basis.shape[0]
    n = order // 2
    rows = np.arange(1, order + 1, dtype=float)[:, None]
    first_columns = np.arange(1, n + 1, dtype=float)[None, :]
    all_columns = np.arange(1, order + 1, dtype=float)[None, :]
    c_z = np.sum((13.0 * rows + 7.0 * first_columns) * first_basis[:, :n])
    c_q = np.sum((11.0 * rows + 5.0 * all_columns) * spectral_basis)
    return np.array([c_z, c_q], dtype=float)

from itertools import permutations

import numpy as np


_NLA14_SHADOW_LANES = 5
_NLA14_MATCHINGS = tuple(permutations(range(_NLA14_SHADOW_LANES)))


def _nla14_run_candidate(A, polar, candidate_seed):
    """Run all seven earlier stages and return one candidate record."""
    order = polar.shape[0]
    n = order // 2
    packed = sample_positive_imaginary_basis(polar, candidate_seed)
    diagonal = packed[order].real
    real_basis = build_real_symplectic_basis(packed[:order])
    blocks = compute_skew_hamiltonian_blocks(A, real_basis)
    eigensystem = solve_half_size_hermitian_problem(blocks)
    spectral_data = recover_real_spectral_decomposition(
        real_basis,
        eigensystem,
    )
    checksums = compute_basis_checksums(
        real_basis,
        spectral_data[:order],
    )
    return (
        float(diagonal.min()),
        float(np.log(diagonal).sum()),
        float(checksums[0]),
        float(checksums[1]),
        spectral_data[order, :n].copy(),
    )


def _nla14_shadow_seed_grid(seed, order, candidates):
    """Create platform-stable independent uint64 streams for the shadow audit."""
    mask = (1 << 32) - 1
    entropy = [
        int(seed) & mask,
        (int(seed) >> 32) & mask,
        int(order) & mask,
        int(candidates) & mask,
        0x4E4C4131,
        0x34534844,
    ]
    root = np.random.SeedSequence(entropy)
    children = root.spawn(int(candidates) * _NLA14_SHADOW_LANES)
    seeds = np.empty(
        (int(candidates), _NLA14_SHADOW_LANES),
        dtype=np.uint64,
    )
    for flat_index, child in enumerate(children):
        seeds.flat[flat_index] = child.generate_state(1, dtype=np.uint64)[0]
    return seeds


def _nla14_matching_distance(left, right):
    """Return the minimum mean Euclidean cost over all five-lane matchings."""
    best = np.inf
    for matching in _NLA14_MATCHINGS:
        cost = 0.0
        for lane, other_lane in enumerate(matching):
            cost += float(np.linalg.norm(left[lane] - right[other_lane]))
        best = min(best, cost)
    return float(best / _NLA14_SHADOW_LANES)


def _nla14_select_medoid(A, polar, seed, candidates, pool, records):
    """Select a robust finalist from a near-maximin pool."""
    pool = np.asarray(pool, dtype=int)
    if pool.size >= 3:
        seeds = _nla14_shadow_seed_grid(seed, polar.shape[0], candidates)
        raw = np.empty((pool.size, _NLA14_SHADOW_LANES, 4), dtype=float)
        for local_index, candidate_index in enumerate(pool):
            for lane in range(_NLA14_SHADOW_LANES):
                record = _nla14_run_candidate(
                    A,
                    polar,
                    int(seeds[candidate_index, lane]),
                )
                raw[local_index, lane] = record[:4]

        flattened = raw.reshape(-1, raw.shape[-1])
        center = np.median(flattened, axis=0)
        mad = np.median(np.abs(flattened - center), axis=0)
        magnitude = np.maximum(1.0, np.max(np.abs(flattened), axis=0))
        scale = np.maximum(mad, 128.0 * np.finfo(float).eps * magnitude)
        profiles = (raw - center) / scale

        distances = np.zeros((pool.size, pool.size), dtype=float)
        for left in range(pool.size):
            for right in range(left + 1, pool.size):
                distance = _nla14_matching_distance(
                    profiles[left],
                    profiles[right],
                )
                distances[left, right] = distance
                distances[right, left] = distance
        eccentricities = distances.sum(axis=1)
        best_eccentricity = float(eccentricities.min())
        eccentricity_tolerance = (
            256.0
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(eccentricities))))
        )
        pool = pool[
            eccentricities <= best_eccentricity + eccentricity_tolerance
        ]

    log_volumes = np.array([records[index][1] for index in pool])
    best_volume = float(log_volumes.max())
    volume_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(best_volume))
    pool = pool[log_volumes >= best_volume - volume_tolerance]

    scores = np.array([records[index][0] for index in pool])
    finalist_score = float(scores.max())
    best_score = max(record[0] for record in records)
    score_tolerance = 64.0 * np.finfo(float).eps * max(1.0, best_score)
    return int(pool[np.flatnonzero(scores >= finalist_score - score_tolerance)[0]])


def compute_basis_sensitive_skew_scalar(
    A: np.ndarray,
    seed: int = 260812153,
    candidates: int = 5,
) -> float:
    """Reference candidate-ensemble calculation composed from the seven earlier oracles."""
    if isinstance(candidates, (bool, np.bool_)) or not isinstance(
        candidates, (int, np.integer)
    ):
        raise ValueError("candidates must be an integer")
    if int(candidates) < 2:
        raise ValueError("candidates must be at least two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    polar = compute_skew_polar_factor(A)
    order = polar.shape[0]
    n = order // 2

    records = []
    for index in range(int(candidates)):
        packed = sample_positive_imaginary_basis(
            polar,
            int(seed) + index,
        )
        diagonal = packed[order].real
        real_basis = build_real_symplectic_basis(packed[:order])
        blocks = compute_skew_hamiltonian_blocks(A, real_basis)
        eigensystem = solve_half_size_hermitian_problem(blocks)
        spectral_data = recover_real_spectral_decomposition(
            real_basis,
            eigensystem,
        )
        checksums = compute_basis_checksums(
            real_basis,
            spectral_data[:order],
        )
        records.append(
            (
                float(diagonal.min()),
                float(np.log(diagonal).sum()),
                float(checksums[0]),
                float(checksums[1]),
                spectral_data[order, :n].copy(),
            )
        )

    scores = np.array([record[0] for record in records])
    z_sums = np.array([record[2] for record in records])
    q_sums = np.array([record[3] for record in records])
    best_score = float(scores.max())
    pool = np.flatnonzero(scores >= 0.95 * best_score)
    selected = _nla14_select_medoid(
        A,
        polar,
        int(seed),
        int(candidates),
        pool,
        records,
    )
    spread = float(q_sums.max() - q_sums.min())
    indices = np.arange(1, n + 1, dtype=float)
    result = float(
        indices @ records[selected][4]
        + 1e-3 * z_sums[selected]
        + 1e-4 * q_sums[selected]
        + 1e-2 * spread
    )
    if not np.isfinite(result):
        raise ValueError("the final scalar is not finite")
    return result
SCICODE_GOLD_EOF

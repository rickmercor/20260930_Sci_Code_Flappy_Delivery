#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def map_atomistic_coordinates(atomistic_positions, mapping):
    import numpy as np

    positions = np.asarray(atomistic_positions, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    if positions.ndim != 3 or min(positions.shape) == 0:
        raise ValueError("atomistic_positions must be a nonempty 3D array")
    if (
        coordinate_map.ndim != 2
        or coordinate_map.shape[0] == 0
        or coordinate_map.shape[1] != positions.shape[1]
    ):
        raise ValueError("mapping has incompatible shape")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(coordinate_map)):
        raise ValueError("positions and mapping must be finite")
    if np.any(coordinate_map < 0):
        raise ValueError("mapping weights must be nonnegative")
    if np.linalg.matrix_rank(coordinate_map) != coordinate_map.shape[0]:
        raise ValueError("mapping must have full row rank")
    if not np.allclose(
        np.sum(coordinate_map, axis=1), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("each mapping row must sum to one")
    return np.einsum("qa,fac->fqc", coordinate_map, positions)

def minimum_variance_force_map(mapping, atomistic_forces, ridge):
    import numpy as np

    coordinate_map = np.asarray(mapping, dtype=float)
    forces = np.asarray(atomistic_forces, dtype=float)
    if forces.ndim != 3 or min(forces.shape) == 0:
        raise ValueError("atomistic_forces must be a nonempty 3D array")
    if (
        coordinate_map.ndim != 2
        or coordinate_map.shape[0] == 0
        or coordinate_map.shape[1] != forces.shape[1]
    ):
        raise ValueError("mapping has incompatible shape")
    if np.any(~np.isfinite(forces)) or np.any(~np.isfinite(coordinate_map)):
        raise ValueError("forces and mapping must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if np.linalg.matrix_rank(coordinate_map) != coordinate_map.shape[0]:
        raise ValueError("mapping must have full row rank")

    observations = np.transpose(forces, (0, 2, 1)).reshape(-1, forces.shape[1])
    second_moment = observations.T @ observations / len(observations)
    regularized = second_moment + ridge * np.eye(forces.shape[1])
    try:
        map_times_inverse = np.linalg.solve(
            regularized, coordinate_map.T
        ).T
        force_map = np.linalg.solve(
            coordinate_map @ map_times_inverse.T,
            map_times_inverse,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("force-map system is singular") from exc
    return force_map

def sample_probabilistic_cg(mapped_positions, standard_normals, sigma):
    import numpy as np

    mapped = np.asarray(mapped_positions, dtype=float)
    normals = np.asarray(standard_normals, dtype=float)
    if mapped.ndim != 3 or min(mapped.shape) == 0:
        raise ValueError("mapped_positions must be a nonempty 3D array")
    if normals.ndim != 4 or normals.shape[1:] != mapped.shape:
        raise ValueError("standard_normals has incompatible shape")
    if np.any(~np.isfinite(mapped)) or np.any(~np.isfinite(normals)):
        raise ValueError("positions and normals must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return mapped[None, ...] + sigma * normals

def build_extended_forces(
    atomistic_forces,
    mapping,
    mapped_positions,
    noised_positions,
    sigma,
):
    import numpy as np

    forces = np.asarray(atomistic_forces, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    mapped = np.asarray(mapped_positions, dtype=float)
    noised = np.asarray(noised_positions, dtype=float)
    if forces.ndim != 3 or coordinate_map.ndim != 2:
        raise ValueError("forces and mapping have invalid rank")
    if mapped.ndim != 3 or noised.ndim != 4:
        raise ValueError("mapped and noised positions have invalid rank")
    if coordinate_map.shape != (mapped.shape[1], forces.shape[1]):
        raise ValueError("mapping has incompatible shape")
    if (
        forces.shape[0] != mapped.shape[0]
        or forces.shape[2] != mapped.shape[2]
        or noised.shape[1:] != mapped.shape
    ):
        raise ValueError("trajectory arrays have incompatible shapes")
    if any(
        np.any(~np.isfinite(value))
        for value in (forces, coordinate_map, mapped, noised)
    ):
        raise ValueError("all arrays must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    displacement = noised - mapped[None, ...]
    inverse_variance = 1.0 / (sigma * sigma)
    atom_block = forces[None, ...] + np.einsum(
        "qa,sfqc->sfac", coordinate_map, displacement
    ) * inverse_variance
    noise_block = -displacement * inverse_variance
    return np.concatenate([atom_block, noise_block], axis=2)

def fit_staged_force_map(joint_forces, base_force_map, ridge):
    import numpy as np

    joint = np.asarray(joint_forces, dtype=float)
    base_map = np.asarray(base_force_map, dtype=float)
    if joint.ndim != 4 or min(joint.shape) == 0:
        raise ValueError("joint_forces must be a nonempty 4D array")
    if base_map.ndim != 2 or min(base_map.shape) == 0:
        raise ValueError("base_force_map must be a nonempty matrix")
    n_beads, n_atoms = base_map.shape
    if joint.shape[2] != n_atoms + n_beads:
        raise ValueError("extended force axis has incompatible length")
    if np.any(~np.isfinite(joint)) or np.any(~np.isfinite(base_map)):
        raise ValueError("joint forces and base map must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    atom_block = joint[:, :, :n_atoms, :]
    noise_block = joint[:, :, n_atoms:, :]
    mapped_atom = np.einsum("qa,sfac->sfqc", base_map, atom_block)
    design = np.transpose(mapped_atom, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    response = np.transpose(noise_block, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    count = len(design)
    gram = design.T @ design / count + ridge * np.eye(n_beads)
    cross = response.T @ design / count
    try:
        stage = -np.linalg.solve(gram.T, cross.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("staged force-map system is singular") from exc
    return np.concatenate([stage @ base_map, np.eye(n_beads)], axis=1)

def project_extended_forces(joint_forces, extended_force_map):
    import numpy as np

    joint = np.asarray(joint_forces, dtype=float)
    force_map = np.asarray(extended_force_map, dtype=float)
    if joint.ndim != 4 or min(joint.shape) == 0:
        raise ValueError("joint_forces must be a nonempty 4D array")
    if (
        force_map.ndim != 2
        or force_map.shape[0] == 0
        or force_map.shape[1] != joint.shape[2]
    ):
        raise ValueError("extended_force_map has incompatible shape")
    if np.any(~np.isfinite(joint)) or np.any(~np.isfinite(force_map)):
        raise ValueError("joint forces and map must be finite")
    return np.einsum("qj,sfjc->sfqc", force_map, joint)

def fit_conservative_precision(noised_positions, target_forces, ridge):
    import numpy as np

    positions = np.asarray(noised_positions, dtype=float)
    targets = np.asarray(target_forces, dtype=float)
    if positions.ndim != 4 or min(positions.shape) == 0:
        raise ValueError("noised_positions must be a nonempty 4D array")
    if targets.shape != positions.shape:
        raise ValueError("target_forces must match noised_positions")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(targets)):
        raise ValueError("positions and targets must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    n_beads = positions.shape[2]
    coordinates = np.transpose(positions, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    responses = np.transpose(targets, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    pairs = [
        (row, column)
        for row in range(n_beads)
        for column in range(row, n_beads)
    ]
    design = np.zeros((len(coordinates) * n_beads, len(pairs)))
    for index, (row, column) in enumerate(pairs):
        if row == column:
            design[row::n_beads, index] = -coordinates[:, row]
        else:
            design[row::n_beads, index] = -coordinates[:, column]
            design[column::n_beads, index] = -coordinates[:, row]

    count = len(coordinates)
    normal = design.T @ design / count + ridge * np.eye(len(pairs))
    right_hand_side = design.T @ responses.reshape(-1) / count
    try:
        parameters = np.linalg.solve(normal, right_hand_side)
    except np.linalg.LinAlgError as exc:
        raise ValueError("precision fit is singular") from exc

    precision = np.zeros((n_beads, n_beads))
    for value, (row, column) in zip(parameters, pairs):
        precision[row, column] = value
        precision[column, row] = value
    return precision

def noised_pmf_precision_error(
    atomistic_precision,
    mapping,
    atomistic_positions,
    atomistic_forces,
    standard_normals,
    sigma,
    force_map_ridge,
    stage_ridge,
    fit_ridge,
):
    import numpy as np

    precision = np.asarray(atomistic_precision, dtype=float)
    positions = np.asarray(atomistic_positions, dtype=float)
    forces = np.asarray(atomistic_forces, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    if (
        precision.ndim != 2
        or precision.shape[0] != precision.shape[1]
        or positions.ndim != 3
        or precision.shape[0] != positions.shape[1]
    ):
        raise ValueError("atomistic precision has incompatible shape")
    if np.any(~np.isfinite(precision)):
        raise ValueError("atomistic precision must be finite")
    if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
        raise ValueError("atomistic precision must be symmetric")
    try:
        np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError("atomistic precision must be positive definite") from exc
    expected_forces = -np.einsum("ab,fbc->fac", precision, positions)
    if forces.shape != positions.shape or not np.allclose(
        forces, expected_forces, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("forces do not match the harmonic reference")

    mapped = map_atomistic_coordinates(positions, coordinate_map)
    base_map = minimum_variance_force_map(
        coordinate_map, forces, force_map_ridge
    )
    noised = sample_probabilistic_cg(
        mapped, standard_normals, sigma
    )
    joint = build_extended_forces(
        forces, coordinate_map, mapped, noised, sigma
    )
    extended_map = fit_staged_force_map(
        joint, base_map, stage_ridge
    )
    targets = project_extended_forces(joint, extended_map)
    fitted = fit_conservative_precision(noised, targets, fit_ridge)

    unnoised_covariance = coordinate_map @ np.linalg.solve(
        precision, coordinate_map.T
    )
    noised_covariance = unnoised_covariance + sigma * sigma * np.eye(
        coordinate_map.shape[0]
    )
    exact = np.linalg.inv(noised_covariance)
    return float(np.linalg.norm(fitted - exact) / np.linalg.norm(exact))


def _draw_harmonic_fixture(seed, precision, mapping, frames, replicas, sigma):
    import numpy as np

    rng = np.random.default_rng(seed)
    n_atoms = len(precision)
    components = 3
    cholesky = np.linalg.cholesky(precision)
    white = rng.normal(size=(frames, components, n_atoms))
    flattened = np.linalg.solve(
        cholesky.T, white.reshape(-1, n_atoms).T
    ).T
    positions = flattened.reshape(frames, components, n_atoms).transpose(
        0, 2, 1
    )
    forces = -np.einsum("ab,fbc->fac", precision, positions)
    normals = rng.normal(
        size=(replicas, frames, components, len(mapping))
    ).transpose(0, 1, 3, 2)
    return (
        precision,
        mapping,
        positions,
        forces,
        normals,
        sigma,
        1e-4,
        0.02,
        1e-5,
    )


def _benchmark_fixture():
    import numpy as np

    fixture_rng = np.random.default_rng(7129)
    matrix = fixture_rng.normal(size=(6, 6))
    precision = matrix.T @ matrix / 6.0 + 1.25 * np.eye(6)
    sites = np.arange(6, dtype=float)
    centers = 0.4 + 2.1 * np.arange(3, dtype=float)
    raw_map = np.exp(
        -0.55 * (sites[None, :] - centers[:, None]) ** 2
        + 0.08 * fixture_rng.normal(size=(3, 6))
    )
    mapping = raw_map / raw_map.sum(axis=1, keepdims=True)
    return _draw_harmonic_fixture(
        20260822, precision, mapping, 8, 16, 0.22
    )


def _one_bead_fixture():
    import numpy as np

    precision = np.array([[2.4, -0.3], [-0.3, 1.8]])
    mapping = np.array([[0.4, 0.6]])
    return _draw_harmonic_fixture(17, precision, mapping, 5, 6, 0.15)


def _overlap_fixture():
    import numpy as np

    precision = np.array(
        [[2.8, -0.5, 0.1], [-0.5, 2.5, -0.4], [0.1, -0.4, 2.1]]
    )
    mapping = np.array([[0.7, 0.3, 0.0], [0.0, 0.35, 0.65]])
    return _draw_harmonic_fixture(311, precision, mapping, 7, 4, 0.18)


def _four_atom_fixture():
    import numpy as np

    precision = np.array(
        [
            [3.2, -0.6, 0.15, 0.0],
            [-0.6, 2.7, -0.35, 0.1],
            [0.15, -0.35, 2.4, -0.45],
            [0.0, 0.1, -0.45, 2.2],
        ]
    )
    mapping = np.array(
        [[0.55, 0.30, 0.15, 0.0], [0.0, 0.10, 0.35, 0.55]]
    )
    return _draw_harmonic_fixture(902, precision, mapping, 6, 5, 0.27)
SCICODE_GOLD_EOF
